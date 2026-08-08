import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const testDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(testDirectory, "..");
const maintainedDocuments = [
  "README.md",
  "RESEARCH_CODE.md",
  "specs/research-code-support.md",
];

async function readMaintainedDocuments() {
  return Promise.all(
    maintainedDocuments.map(async (path) => ({
      path,
      text: await readFile(resolve(repositoryRoot, path), "utf8"),
    })),
  );
}

function localMarkdownTargets(markdown) {
  return [...markdown.matchAll(/\[[^\]]*\]\(([^)]+)\)/g)]
    .map(([, target]) => target.trim().replace(/^<|>$/g, ""))
    .filter(
      (target) =>
        target &&
        !target.startsWith("#") &&
        !target.startsWith("http://") &&
        !target.startsWith("https://") &&
        !target.startsWith("mailto:"),
    )
    .map((target) => decodeURIComponent(target.split("#", 1)[0]));
}

function pythonScriptCommands(path, text) {
  return [
    ...text.matchAll(
      /^\s*(?:[$>]\s*)?(python(?:3)?)\s+([^\s]+\.py)(?:\s|$)/gm,
    ),
  ].map(([, executable, script]) => ({ executable, path, script }));
}

test("maintained documentation has no broken local links", async () => {
  const documents = await readMaintainedDocuments();

  await Promise.all(
    documents.flatMap(({ path, text }) =>
      localMarkdownTargets(text).map(async (target) => {
        const documentDirectory = dirname(resolve(repositoryRoot, path));
        const absoluteTarget = resolve(documentDirectory, target);
        const repositoryRelativeTarget = relative(repositoryRoot, absoluteTarget);

        assert.equal(
          repositoryRelativeTarget === ".." ||
            repositoryRelativeTarget.startsWith(`..${sep}`) ||
            isAbsolute(repositoryRelativeTarget),
          false,
          `${path} link escapes the repository: ${target}`,
        );

        await assert.doesNotReject(
          access(absoluteTarget),
          `${path} links to missing local path: ${target}`,
        );
      }),
    ),
  );
});

test("documented Python commands only name checked-in scripts", async () => {
  const documents = await readMaintainedDocuments();
  const commands = documents.flatMap(({ path, text }) =>
    pythonScriptCommands(path, text));

  for (const { path, script } of commands) {
    await assert.doesNotReject(
      access(resolve(repositoryRoot, script)),
      `${path} documents a missing Python script: ${script}`,
    );
  }
});

test("archival research entry points are not documented as runnable commands", async () => {
  const documents = await readMaintainedDocuments();
  const commands = documents.flatMap(({ path, text }) =>
    pythonScriptCommands(path, text));
  const unsupportedEntrypoints = new Set(
    ["train.py", "eval.py", "surprisenet_train.py", "surprisenet_inference.py"]
      .map((script) => resolve(repositoryRoot, script)),
  );

  for (const { path, script } of commands) {
    assert.equal(
      unsupportedEntrypoints.has(resolve(repositoryRoot, script)),
      false,
      `${path} presents an unsupported research artifact as runnable: ${script}`,
    );
  }
});

test("README keeps the research-code warning concise and visitor-facing", async () => {
  const readme = await readFile(resolve(repositoryRoot, "README.md"), "utf8");

  assert.match(
    readme,
    /precomputed samples\. It does not run model inference and does not require a backend\./i,
  );
  assert.match(readme, /RESEARCH_CODE\.md/);
  assert.match(
    readme,
    /does not provide reproducible preprocessing, supported end-to-end training or custom-melody inference, or a pretrained checkpoint\./i,
  );
  assert.doesNotMatch(
    readme,
    /localhost|http\.server|node --test|GitHub Actions|repository settings/i,
  );
  assert.doesNotMatch(readme, /python\s+surprisenet_(?:train|inference)\.py/i);
});

test("web demo CI checks pull requests without attempting a Pages deployment", async () => {
  const workflow = await readFile(
    resolve(repositoryRoot, ".github/workflows/pages.yml"),
    "utf8",
  );

  assert.match(workflow, /pull_request:\s*\n\s+branches:\s*\n\s+- master/);
  assert.match(workflow, /push:\s*\n\s+branches:\s*\n\s+- master/);
  assert.match(workflow, /actions\/checkout@v\d+/);
  assert.match(workflow, /actions\/setup-node@v\d+/);
  assert.match(workflow, /node-version:\s*["']?24["']?/);
  assert.match(
    workflow,
    /node --test tests\/repository-docs\.test\.mjs tests\/web-demo\.test\.mjs/,
  );

  assert.doesNotMatch(
    workflow,
    /configure-pages|upload-pages-artifact|deploy-pages|pages:\s*write|id-token:\s*write/,
  );
});
