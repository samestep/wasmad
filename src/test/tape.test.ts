import binaryen from "binaryen";
import { expect, test } from "vitest";
import { Tape, tape } from "../tape.js";
import * as util from "../util.js";
import { slurp } from "./util.js";

const empty = (): Tape => {
  const [struct] = util.buildType(1, (builder) => {
    builder.setStructType(0, []);
  });
  return { fwd: [], bwd: new Map(), struct };
};

test("get param", () => {
  const mod = new binaryen.Module();
  try {
    mod.addFunction(
      "foo",
      binaryen.f64,
      binaryen.f64,
      [],
      mod.f64.mul(
        mod.local.get(0, binaryen.f64),
        mod.local.get(0, binaryen.f64),
      ),
    );
    expect(tape(mod)).toEqual([empty()]);
  } finally {
    mod.dispose();
  }
});

test("set and get param", () => {
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

test("square, set, and get", async () => {
  const mod = binaryen.parseText(await slurp("square.wat"));
  try {
    expect(tape(mod)).toEqual([empty()]);
  } finally {
    mod.dispose();
  }
});
