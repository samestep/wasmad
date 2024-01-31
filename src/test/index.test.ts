import binaryen from "binaryen";
import fs from "fs/promises";
import path from "path";
import url from "url";
import { expect, test } from "vitest";

const compile = async <T extends WebAssembly.Exports>(
  filename: string,
): Promise<T> => {
  const dir = path.dirname(url.fileURLToPath(import.meta.url));
  const text = await fs.readFile(path.join(dir, filename), "utf8");
  const module = binaryen.parseText(text);
  const binary = module.emitBinary();
  module.dispose();
  const compiled = await WebAssembly.compile(binary);
  const instance = await WebAssembly.instantiate(compiled);
  return instance.exports as T;
};

test("division", async () => {
  const { div } = await compile<{ div: (a: number, b: number) => number }>(
    "div.wat",
  );
  expect(div(2, 3)).toBe(2 / 3);
});

test("multiple memories", async () => {
  const { store, div } = await compile<{
    store: (a: number, b: number) => void;
    div: () => number;
  }>("multi-memory.wat");
  store(2, 3);
  expect(div()).toBe(2 / 3);
});
