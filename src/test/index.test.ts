import binaryen from "binaryen";
import fs from "fs/promises";
import path from "path";
import url from "url";
import { expect, test } from "vitest";
import { autodiff } from "../index.js";

const dir = path.dirname(url.fileURLToPath(import.meta.url));

const slurp = async (filename: string): Promise<string> =>
  await fs.readFile(path.join(dir, filename), "utf8");

const wat = async (text: string): Promise<Uint8Array> => {
  const mod = binaryen.parseText(text);
  try {
    return mod.emitBinary();
  } finally {
    mod.dispose();
  }
};

const compile = async <T extends WebAssembly.Exports>(
  binary: Uint8Array,
): Promise<T> => {
  const compiled = await WebAssembly.compile(binary);
  const instance = await WebAssembly.instantiate(compiled);
  return instance.exports as T;
};

type Binop = {
  fwd: (
    a: number,
    b: number,
    da: number,
    db: number,
  ) => [number, number, number];
  bwd: (
    a: number,
    b: number,
    da: number,
    db: number,
    c: number,
    dc: number,
    t: number,
  ) => [number, number];
};

const binop = async (filename: string): Promise<Binop> => {
  let binary;
  const mod = binaryen.parseText(await slurp(filename));
  try {
    mod.setFeatures(binaryen.Features.Multivalue);
    autodiff(mod);
    binary = mod.emitBinary();
  } finally {
    mod.dispose();
  }
  return await compile<Binop>(binary);
};

test("subtraction", async () => {
  const { fwd, bwd } = await binop("sub.wat");
  const a = 5;
  const b = 3;
  let da = 0;
  let db = 0;
  let [c, dc, t] = fwd(a, b, da, db);
  expect([c, dc, t]).toEqual([2, 0, 0]);
  dc = 1;
  [da, db] = bwd(a, b, da, db, c, dc, t);
  expect([da, db]).toEqual([1, -1]);
});

test("division", async () => {
  const { fwd, bwd } = await binop("div.wat");
  const a = 5;
  const b = 3;
  let da = 0;
  let db = 0;
  let [c, dc, t] = fwd(a, b, da, db);
  expect([c, dc, t]).toEqual([5 / 3, 0, 0]);
  dc = 1;
  [da, db] = bwd(a, b, da, db, c, dc, t);
  expect([da, db]).toEqual([1 / 3, -5 / 9]);
});

test("square", async () => {
  const { square } = await compile<{ square: (x: number) => number }>(
    await wat(await slurp("square.wat")),
  );
  expect(square(3)).toBe(9);
});

test("polynomial", async () => {
  const { polynomial } = await compile<{
    polynomial: (x: number, y: number) => number;
  }>(await wat(await slurp("polynomial.wat")));
  const x = 2;
  const y = 3;
  expect(polynomial(x, y)).toBe(
    2 * x ** 3 + 4 * x ** 2 * y + x * y ** 5 + y ** 2 - 7,
  );
});

test("multiple memories", async () => {
  const { store, div } = await compile<{
    store: (a: number, b: number) => void;
    div: () => number;
  }>(await wat(await slurp("multi-memory.wat")));
  store(2, 3);
  expect(div()).toBe(2 / 3);
});
