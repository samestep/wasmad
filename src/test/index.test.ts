import binaryen from "binaryen";
import path from "path";
import { beforeAll, expect, test } from "vitest";
import * as wasmad from "../index.js";
import * as util from "../util.js";
import { dir, mkdirp, rmrf, slurp, spit } from "./util.js";

const out = "out";

beforeAll(async () => {
  await rmrf(out);
  await mkdirp(out);
});

const features =
  binaryen.Features.Multivalue |
  binaryen.Features.ReferenceTypes |
  binaryen.Features.GC;

const wat = async (text: string): Promise<Uint8Array> => {
  const mod = binaryen.parseText(text);
  try {
    mod.setFeatures(features);
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

interface AD {
  orig?: WebAssembly.Exports;
  fwd?: WebAssembly.Exports;
  bwd?: WebAssembly.Exports;
}

interface Unop<K extends string> extends AD {
  fwd: { [k in K]: (a: number, da: number) => [number, number, any] };
  bwd: { [k in K]: (da: number, db: number, t: any) => [number, number] };
}

interface Binop<K extends string> extends AD {
  fwd: {
    [k in K]: (
      a: number,
      b: number,
      da: number,
      db: number,
    ) => [number, number, any];
  };
  bwd: {
    [k in K]: (da: number, db: number, dc: number, t: any) => [number, number];
  };
}

interface Names {
  name: string;
  orig: string;
  fwd: string;
  bwd: string;
}

const autodiff = async <T extends AD>(filename: string): Promise<T> => {
  const mod = binaryen.parseText(await slurp(filename));
  try {
    mod.setFeatures(features);

    const set = new util.Names();
    const n = mod.getNumExports();
    for (let i = 0; i < n; ++i)
      set.add(binaryen.getExportInfo(mod.getExportByIndex(i)).name);

    const names = wasmad.autodiff(mod).map((f, i): Names => {
      const name = binaryen.getFunctionInfo(mod.getFunctionByIndex(i)).name;
      const fwdName = binaryen.getFunctionInfo(f.fwd).name;
      const bwdName = binaryen.getFunctionInfo(f.bwd).name;
      const orig = set.make(name);
      const fwd = set.make(fwdName);
      const bwd = set.make(bwdName);
      mod.addFunctionExport(name, orig);
      mod.addFunctionExport(fwdName, fwd);
      mod.addFunctionExport(bwdName, bwd);
      return { name, orig, fwd, bwd };
    });

    const filepath = path.join(out, filename);
    await spit(filepath, mod.emitText());
    if (!mod.validate())
      throw Error(`invalid module: ${path.join(dir, filepath)}`);

    const binary = mod.emitBinary();
    const exports = await compile<any>(binary);

    const origs: WebAssembly.Exports = {};
    const fwds: WebAssembly.Exports = {};
    const bwds: WebAssembly.Exports = {};
    for (const { name, orig, fwd, bwd } of names) {
      origs[name] = exports[orig];
      fwds[name] = exports[fwd];
      bwds[name] = exports[bwd];
    }
    return { orig: origs, fwd: fwds, bwd: bwds } as T;
  } finally {
    mod.dispose();
  }
};

test("subtraction", async () => {
  const { fwd, bwd } = await autodiff<Binop<"sub">>("sub.wat");
  let da = 0;
  let db = 0;
  let [c, dc, t] = fwd.sub(5, 3, da, db);
  expect([c, dc]).toEqual([2, 0]);
  dc = 1;
  [da, db] = bwd.sub(da, db, dc, t);
  expect([da, db]).toEqual([dc, -dc]);
});

test("division", async () => {
  const { fwd, bwd } = await autodiff<Binop<"div">>("div.wat");
  const a = 5;
  const b = 3;
  let da = 0;
  let db = 0;
  let [c, dc, t] = fwd.div(a, b, da, db);
  expect([c, dc]).toEqual([a / b, 0]);
  dc = 1;
  [da, db] = bwd.div(da, db, dc, t);
  expect([da, db]).toEqual([dc / b, dc * (-a / b ** 2)]);
});

test("square", async () => {
  const { fwd, bwd } = await autodiff<Unop<"sqr">>("square.wat");
  const x = 3;
  let dx = 0;
  let [y, dy, t] = fwd.sqr(x, dx);
  expect([y, dy]).toEqual([9, 0]);
  dx = 5;
  dy = 1;
  expect(bwd.sqr(dx, dy, t)).toBe(dx + dy * 2 * x);
});

test("tesseract", async () => {
  const { fwd, bwd } = await autodiff<Unop<"f">>("tesseract.wat");
  const x = 5;
  const dx = 0;
  const [y, dy, t] = fwd.f(x, dx);
  expect([y, dy]).toEqual([x ** 4, 0]);
  expect(bwd.f(dx, 1, t)).toBe(4 * x ** 3);
});

test("polynomial", async () => {
  const { fwd, bwd } = await autodiff<Binop<"f">>("polynomial.wat");
  const x = 2;
  const y = 2;
  const dx = 0;
  const dy = 0;
  const [z, dz, t] = fwd.f(x, y, dx, dy);
  expect([z, dz]).toEqual([
    2 * x ** 3 + 4 * x ** 2 * y + x * y ** 5 + y ** 2 - 7,
    0,
  ]);
  expect(bwd.f(dx, dy, 1, t)).toEqual([
    6 * x ** 2 + 8 * x * y + y ** 5,
    4 * x ** 2 + 5 * x * y ** 4 + 2 * y,
  ]);
});

test("composition", async () => {
  const { fwd, bwd } = await autodiff<Unop<"gf" | "fg">>("compose.wat");
  const x = 5;

  let [y, , t] = fwd.gf(x, 0);
  expect(y).toBe(x ** 2 + 1);
  expect(bwd.gf(0, 1, t)).toBe(2 * x);

  [y, , t] = fwd.fg(x, 0);
  expect(y).toBe((x + 1) ** 2);
  expect(bwd.fg(0, 1, t)).toBe(2 * (x + 1));
});

test("arrays", async () => {
  type I32 = number;
  type F64 = number;

  type I32s = any;
  type F64s = any;
  type I32ss = any;
  type F64ss = any;

  const {
    i32sNew,
    f64sNew,
    i32ssNew,
    f64ssNew,

    i32sLen,
    f64sLen,
    i32ssLen,
    f64ssLen,

    i32sGet,
    f64sGet,
    i32ssGet,
    f64ssGet,

    i32sSet,
    f64sSet,
    i32ssSet,
    f64ssSet,

    f,
  } = await compile<{
    i32sNew: (n: I32) => I32s;
    f64sNew: (n: I32) => F64s;
    i32ssNew: (n: I32) => I32ss;
    f64ssNew: (n: I32) => F64ss;

    i32sLen: (a: null | I32s) => I32;
    f64sLen: (a: null | F64s) => I32;
    i32ssLen: (a: I32ss) => I32;
    f64ssLen: (a: F64ss) => I32;

    i32sGet: (a: null | I32s, i: I32) => I32;
    f64sGet: (a: null | F64s, i: I32) => F64;
    i32ssGet: (a: I32ss, i: I32) => null | I32s;
    f64ssGet: (a: F64ss, i: I32) => null | F64s;

    i32sSet: (a: null | I32s, i: I32, x: I32) => void;
    f64sSet: (a: null | F64s, i: I32, x: F64) => void;
    i32ssSet: (a: I32s, i: I32, x: null | I32s) => void;
    f64ssSet: (a: F64s, i: I32, x: null | F64s) => void;

    f: (i: I32, j: I32, iss: I32ss, jss: I32ss, xss: F64ss, yss: F64ss) => F64;
  }>(await wat(await slurp("array.wat")));

  const i32sWrite = (a: I32[]): I32s => {
    const b = i32sNew(a.length);
    a.forEach((x, i) => i32sSet(b, i, x));
    return b;
  };
  const f64sWrite = (a: F64[]): F64s => {
    const b = f64sNew(a.length);
    a.forEach((x, i) => f64sSet(b, i, x));
    return b;
  };
  const i32ssWrite = (a: I32[][]): I32ss => {
    const b = i32ssNew(a.length);
    a.forEach((x, i) => i32ssSet(b, i, i32sWrite(x)));
    return b;
  };
  const f64ssWrite = (a: F64[][]): F64ss => {
    const b = f64ssNew(a.length);
    a.forEach((x, i) => f64ssSet(b, i, f64sWrite(x)));
    return b;
  };

  const i32sRead = (a: I32s): I32[] => {
    const n = i32sLen(a);
    const b = [];
    for (let i = 0; i < n; ++i) b.push(i32sGet(a, i));
    return b;
  };
  const f64sRead = (a: F64s): F64[] => {
    const n = f64sLen(a);
    const b = [];
    for (let i = 0; i < n; ++i) b.push(f64sGet(a, i));
    return b;
  };
  const i32ssRead = (a: I32ss): I32[][] => {
    const n = i32ssLen(a);
    const b = [];
    for (let i = 0; i < n; ++i) b.push(i32sRead(i32ssGet(a, i)));
    return b;
  };
  const f64ssRead = (a: F64ss): F64[][] => {
    const n = f64ssLen(a);
    const b = [];
    for (let i = 0; i < n; ++i) b.push(f64sRead(f64ssGet(a, i)));
    return b;
  };

  const iss = i32ssWrite([
    [0, 3, 1, 0],
    [3, 1, 2, 3],
    [2, 2, 1, 3],
    [2, 0, 0, 1],
  ]);
  const jss = i32ssWrite([
    [3, 0, 0, 3],
    [0, 3, 1, 2],
    [2, 1, 3, 2],
    [1, 0, 1, 2],
  ]);
  const xss = f64ssWrite([
    [89, 3, 61, 107],
    [17, 13, 2, 101],
    [37, 71, 109, 31],
    [43, 19, 67, 7],
  ]);
  const yss = f64ssWrite([
    [47, 113, 5, 79],
    [41, 29, 23, 59],
    [83, 11, 103, 131],
    [127, 73, 97, 53],
  ]);

  expect(f(3, 0, iss, jss, xss, yss)).toBe(23 / 2);

  expect(i32ssRead(iss)).toEqual([
    [1, 0, 1, 1],
    [3, 1, 2, 3],
    [2, 2, 1, 3],
    [2, 0, 0, 2],
  ]);
  expect(i32ssRead(jss)).toEqual([
    [2, 0, 0, 2],
    [0, 3, 1, 2],
    [2, 1, 3, 2],
    [1, 0, 1, 1],
  ]);
  expect(f64ssRead(xss)).toEqual([
    [89, 3, 61, 107],
    [17, 2, 2, 101],
    [41, 23, 23, 59],
    [43, 19, 67, 7],
  ]);
  expect(f64ssRead(yss)).toEqual([
    [47, 113, 5, 79],
    [41, 23, 23, 59],
    [17, 2, 2, 101],
    [127, 73, 97, 53],
  ]);
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
