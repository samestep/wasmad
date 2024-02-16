import binaryen from "binaryen";
import { describe, expect, test } from "vitest";
import * as wasmad from "../index.js";
import * as util from "../util.js";
import { slurp } from "./util.js";

const wat = async (text: string): Promise<Uint8Array> => {
  const mod = binaryen.parseText(text);
  try {
    mod.setFeatures(binaryen.Features.GC);
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

type Unop = {
  fwd: (a: number, da: number) => [number, number, any];
  bwd: (da: number, db: number, t: any) => [number, number];
};

type Binop = {
  fwd: (a: number, b: number, da: number, db: number) => [number, number, any];
  bwd: (da: number, db: number, dc: number, t: any) => [number, number];
};

const autodiff = async <T extends WebAssembly.Exports>(
  filename: string,
  funcname: string,
): Promise<T> => {
  const mod = binaryen.parseText(await slurp(filename));
  try {
    const i = util.unwrap(util.funcIndicesByName(mod).get(funcname));
    mod.setFeatures(binaryen.Features.Multivalue);
    mod.setFeatures(binaryen.Features.GC);
    const { fwd, bwd } = wasmad.autodiff(mod)[i];
    mod.addFunctionExport(binaryen.getFunctionInfo(fwd).name, "fwd");
    mod.addFunctionExport(binaryen.getFunctionInfo(bwd).name, "bwd");
    const binary = mod.emitBinary();
    return await compile<T>(binary);
  } finally {
    mod.dispose();
  }
};

test("subtraction", async () => {
  const { fwd, bwd } = await autodiff<Binop>("sub.wat", "sub");
  let da = 0;
  let db = 0;
  let [c, dc, t] = fwd(5, 3, da, db);
  expect([c, dc]).toEqual([2, 0]);
  dc = 1;
  [da, db] = bwd(da, db, dc, t);
  expect([da, db]).toEqual([dc, -dc]);
});

test("division", async () => {
  const { fwd, bwd } = await autodiff<Binop>("div.wat", "div");
  const a = 5;
  const b = 3;
  let da = 0;
  let db = 0;
  let [c, dc, t] = fwd(a, b, da, db);
  expect([c, dc]).toEqual([a / b, 0]);
  dc = 1;
  [da, db] = bwd(da, db, dc, t);
  expect([da, db]).toEqual([dc / b, dc * (-a / b ** 2)]);
});

test("square", async () => {
  const { fwd, bwd } = await autodiff<Unop>("square.wat", "sqr");
  const x = 3;
  let dx = 0;
  let [y, dy, t] = fwd(x, dx);
  expect([y, dy]).toEqual([9, 0]);
  dx = 5;
  dy = 1;
  expect(bwd(dx, dy, t)).toBe(dx + dy * 2 * x);
});

test("tesseract", async () => {
  const { fwd, bwd } = await autodiff<Unop>("tesseract.wat", "f");
  const x = 5;
  const dx = 0;
  const [y, dy, t] = fwd(x, dx);
  expect([y, dy]).toEqual([x ** 4, 0]);
  expect(bwd(dx, 1, t)).toBe(4 * x ** 3);
});

test("polynomial", async () => {
  const { fwd, bwd } = await autodiff<Binop>("polynomial.wat", "f");
  const x = 2;
  const y = 2;
  const dx = 0;
  const dy = 0;
  const [z, dz, t] = fwd(x, y, dx, dy);
  expect([z, dz]).toEqual([
    2 * x ** 3 + 4 * x ** 2 * y + x * y ** 5 + y ** 2 - 7,
    0,
  ]);
  expect(bwd(dx, dy, 1, t)).toEqual([
    6 * x ** 2 + 8 * x * y + y ** 5,
    4 * x ** 2 + 5 * x * y ** 4 + 2 * y,
  ]);
});

describe("composition", () => {
  const x = 5;

  test("g ∘ f", async () => {
    const { fwd, bwd } = await autodiff<Unop>("compose.wat", "gf");
    const [y, , t] = fwd(x, 0);
    expect(y).toBe(x ** 2 + 1);
    expect(bwd(0, 1, t)).toBe(2 * x);
  });

  test("f ∘ g", async () => {
    const { fwd, bwd } = await autodiff<Unop>("compose.wat", "fg");
    const [y, , t] = fwd(x, 0);
    expect(y).toBe((x + 1) ** 2);
    expect(bwd(0, 1, t)).toBe(2 * (x + 1));
  });
});

test("garbage collection", async () => {
  const { cons, div } = await compile<{
    cons: (a: number, b: number) => any;
    div: (ab: any) => number;
  }>(await wat(await slurp("gc.wat")));
  expect(div(cons(2, 3))).toBe(2 / 3);
});

test("tail call", async () => {
  const { fac } = await compile<{
    fac: (x: bigint) => bigint;
  }>(await wat(await slurp("fac.wat")));
  expect(fac(5n)).toBe(120n);
});
