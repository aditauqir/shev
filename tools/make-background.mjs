#!/usr/bin/env node
// Turn a raster image into `assets/background.rgb`, the wallpaper shev embeds.
//
// shev is a terminal app, so the background is stored as raw RGB pixels and
// drawn as truecolour half-blocks (two pixels per cell). That keeps the photo's
// own colours, which the palette deliberately does not touch.
//
//   node tools/make-background.mjs <image> [width] [height]
//
// `sips` (macOS) does the decode + resample and hands us a BMP, which is trivial
// to read. The file we write is:
//
//   u16 width | u16 height | width * height * 3 bytes of RGB, row-major
//
// height must be even: each terminal cell is two vertical pixels.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const [, , image, widthArg, heightArg] = process.argv;
if (!image) {
  console.error("usage: node tools/make-background.mjs <image> [width] [height]");
  process.exit(1);
}

const width = Number(widthArg) || 160;
let height = Number(heightArg) || 108;
if (height % 2 !== 0) height += 1; // two pixels per cell

const tmp = mkdtempSync(join(tmpdir(), "shev-bg-"));
const bmp = join(tmp, "background.bmp");
execFileSync("sips", [
  "-s", "format", "bmp",
  "--resampleHeightWidth", String(height), String(width),
  image,
  "--out", bmp,
], { stdio: ["ignore", "ignore", "inherit"] });

const buf = readFileSync(bmp);
if (buf.toString("ascii", 0, 2) !== "BM") throw new Error("not a BMP");
const dataOffset = buf.readUInt32LE(10);
const headerSize = buf.readUInt32LE(14);
const w = buf.readInt32LE(18);
const rawHeight = buf.readInt32LE(22);
const bpp = buf.readUInt16LE(28);
const topDown = rawHeight < 0;
const h = Math.abs(rawHeight);
const bytesPerPixel = bpp / 8;
const rowSize = Math.floor((bpp * w + 31) / 32) * 4;
if (bpp !== 24 && bpp !== 32) throw new Error(`unsupported bpp ${bpp}`);
void headerSize;

const rgb = Buffer.alloc(w * h * 3);
for (let y = 0; y < h; y++) {
  const srcY = topDown ? y : h - 1 - y;
  const row = dataOffset + srcY * rowSize;
  for (let x = 0; x < w; x++) {
    const p = row + x * bytesPerPixel;
    const out = (y * w + x) * 3;
    rgb[out] = buf[p + 2];     // BMP stores BGR(A)
    rgb[out + 1] = buf[p + 1];
    rgb[out + 2] = buf[p];
  }
}

const header = Buffer.alloc(4);
header.writeUInt16LE(w, 0);
header.writeUInt16LE(h, 2);

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "..", "assets", "background.rgb");
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, Buffer.concat([header, rgb]));
console.log(`wrote ${out} (${w}x${h}, ${header.length + rgb.length} bytes)`);
