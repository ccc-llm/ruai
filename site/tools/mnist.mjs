// Make site/src/data/mnist.bin.gz, the MNIST subset of the MNIST demo, from the
// four files of the MNIST distribution (train-images-idx3-ubyte.gz, ...):
//   node site/tools/mnist.mjs DIR [TRAIN=12000] [TEST=2000]
// The file (gzip) is: "MNST", u32 train count, u32 test count (little-endian),
// then the train images (28 × 28 bytes each), the test images, the train
// labels and the test labels. The first TRAIN / TEST images of each set are kept.
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync, gzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const [dir, nTrain = "12000", nTest = "2000"] = process.argv.slice(2);
if (!dir) {
  console.error("usage: node site/tools/mnist.mjs DIR [TRAIN] [TEST]");
  process.exit(2);
}
const idx = (name, magic) => {
  const b = gunzipSync(readFileSync(join(dir, name)));
  if (b.readUInt32BE(0) !== magic) throw new Error(`${name}: not an idx file`);
  return b;
};
const images = (name, n) => {
  const b = idx(name, 2051);
  if (b.readUInt32BE(8) !== 28 || b.readUInt32BE(12) !== 28) throw new Error(`${name}: not 28 × 28`);
  if (n > b.readUInt32BE(4)) throw new Error(`${name}: has only ${b.readUInt32BE(4)} images`);
  return b.subarray(16, 16 + n * 784);
};
const labels = (name, n) => idx(name, 2049).subarray(8, 8 + n);

const tr = +nTrain, te = +nTest;
const head = Buffer.alloc(12);
head.write("MNST", 0, "latin1");
head.writeUInt32LE(tr, 4);
head.writeUInt32LE(te, 8);
const body = Buffer.concat([
  head,
  images("train-images-idx3-ubyte.gz", tr),
  images("t10k-images-idx3-ubyte.gz", te),
  labels("train-labels-idx1-ubyte.gz", tr),
  labels("t10k-labels-idx1-ubyte.gz", te),
]);
const out = join(dirname(fileURLToPath(import.meta.url)), "../src/data/mnist.bin.gz");
writeFileSync(out, gzipSync(body, { level: 9 }));
console.log(`${out}: ${tr} train + ${te} test images, ${(gzipSync(body, { level: 9 }).length / 1024).toFixed(0)} KB`);
