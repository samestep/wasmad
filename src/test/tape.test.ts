import binaryen from "binaryen";
import { expect, test } from "vitest";
import { LoadKind, makeTapes } from "../tape.js";
import { Types } from "../type.js";
import * as util from "../util.js";

test("get param", () => {
  const mod = new binaryen.Module();
  try {
    const firstGet = mod.local.get(0, binaryen.f64);
    const secondGet = mod.local.get(0, binaryen.f64);
    const mul = mod.f64.mul(firstGet, secondGet);
    mod.addFunction("foo", binaryen.f64, binaryen.f64, [], mul);
    expect(makeTapes(new Types(), mod)).toEqual([
      {
        fields: 1,
        stores: new Map([[firstGet, 0]]),
        grads: new Map(),
        sets: new Map(),
        calls: new Map(),
        loads: new Map([
          [firstGet, { kind: LoadKind.Field, index: 0 }],
          [secondGet, { kind: LoadKind.Field, index: 0 }],
        ]),
        gradLoads: new Map(),
        struct: util.buildStructType([
          {
            type: binaryen.f64,
            packedType: util.packedTypeNotPacked,
            mutable: false,
          },
        ]),
      },
    ]);
  } finally {
    mod.dispose();
  }
});

test("nonzero constant", () => {
  const mod = new binaryen.Module();
  try {
    const fortyTwo = mod.f64.const(42);
    const tee = mod.local.tee(0, fortyTwo, binaryen.f64);
    const get = mod.local.get(0, binaryen.f64);
    const mul = mod.f64.mul(tee, get);
    mod.addFunction("foo", binaryen.f64, binaryen.f64, [], mul);
    expect(makeTapes(new Types(), mod)).toEqual([
      {
        fields: 0,
        stores: new Map(),
        grads: new Map(),
        sets: new Map(),
        calls: new Map(),
        loads: new Map([
          [tee, { kind: LoadKind.Const, value: 42 }],
          [get, { kind: LoadKind.Const, value: 42 }],
        ]),
        gradLoads: new Map(),
        struct: util.buildStructType([]),
      },
    ]);
  } finally {
    mod.dispose();
  }
});

test("division", () => {
  const mod = new binaryen.Module();
  try {
    const get0 = mod.local.get(0, binaryen.f64);
    const get1 = mod.local.get(1, binaryen.f64);
    const div = mod.f64.div(get0, get1);
    mod.addFunction(
      "foo",
      binaryen.createType([binaryen.f64, binaryen.f64]),
      binaryen.f64,
      [],
      div,
    );
    expect(makeTapes(new Types(), mod)).toEqual([
      {
        fields: 2,
        stores: new Map([
          [get1, 0],
          [div, 1],
        ]),
        grads: new Map(),
        sets: new Map(),
        calls: new Map(),
        loads: new Map([
          [get1, { kind: LoadKind.Field, index: 0 }],
          [div, { kind: LoadKind.Field, index: 1 }],
        ]),
        gradLoads: new Map(),
        struct: util.buildStructType([
          {
            type: binaryen.f64,
            packedType: util.packedTypeNotPacked,
            mutable: false,
          },
          {
            type: binaryen.f64,
            packedType: util.packedTypeNotPacked,
            mutable: false,
          },
        ]),
      },
    ]);
  } finally {
    mod.dispose();
  }
});

test("get unset variable", () => {
  const mod = new binaryen.Module();
  try {
    const get0 = mod.local.get(0, binaryen.f64);
    const get1 = mod.local.get(1, binaryen.f64);
    const mul = mod.f64.mul(get0, get1);
    mod.addFunction("foo", binaryen.f64, binaryen.f64, [binaryen.f64], mul);
    expect(makeTapes(new Types(), mod)).toEqual([
      {
        fields: 1,
        stores: new Map([[get0, 0]]),
        grads: new Map(),
        sets: new Map(),
        calls: new Map(),
        loads: new Map([
          [get0, { kind: LoadKind.Field, index: 0 }],
          [get1, { kind: LoadKind.Const, value: 0 }],
        ]),
        gradLoads: new Map(),
        struct: util.buildStructType([
          {
            type: binaryen.f64,
            packedType: util.packedTypeNotPacked,
            mutable: false,
          },
        ]),
      },
    ]);
  } finally {
    mod.dispose();
  }
});
