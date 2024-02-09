import binaryen from "binaryen";
import { expect, test } from "vitest";
import { tape } from "../tape.js";
import * as util from "../util.js";

test("get param", () => {
  const mod = new binaryen.Module();
  try {
    const firstGet = mod.local.get(0, binaryen.f64);
    const secondGet = mod.local.get(0, binaryen.f64);
    const mul = mod.f64.mul(firstGet, secondGet);
    mod.addFunction("foo", binaryen.f64, binaryen.f64, [], mul);
    const [struct] = util.buildType(1, (builder) => {
      builder.setStructType(0, [
        {
          type: binaryen.f64,
          packedType: util.packedTypeNotPacked,
          mutable: false,
        },
      ]);
    });
    expect(tape(mod)).toEqual([
      {
        fwd: [firstGet],
        bwd: new Map([
          [firstGet, 0],
          [secondGet, 0],
        ]),
        struct,
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
    const [struct] = util.buildType(1, (builder) => {
      builder.setStructType(0, [
        {
          type: binaryen.f64,
          packedType: util.packedTypeNotPacked,
          mutable: false,
        },
      ]);
    });
    expect(tape(mod)).toEqual([
      {
        fwd: [fortyTwo],
        bwd: new Map([
          [tee, 0],
          [get, 0],
        ]),
        struct,
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
    const [struct] = util.buildType(1, (builder) => {
      builder.setStructType(0, [
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
      ]);
    });
    expect(tape(mod)).toEqual([
      {
        fwd: [get1, div],
        bwd: new Map([
          [get1, 0],
          [div, 1],
        ]),
        struct,
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
    const [struct] = util.buildType(1, (builder) => {
      builder.setStructType(0, [
        {
          type: binaryen.f64,
          packedType: util.packedTypeNotPacked,
          mutable: false,
        },
      ]);
    });
    expect(tape(mod)).toEqual([
      {
        fwd: [get0],
        bwd: new Map([[get0, 0]]),
        struct,
      },
    ]);
  } finally {
    mod.dispose();
  }
});
