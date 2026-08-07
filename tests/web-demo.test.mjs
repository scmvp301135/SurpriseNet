import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  CONTOURS,
  SONGS,
  VIDEO_COUNT,
  getSampleSources,
} from "../docs/static/js/catalog.mjs";

const testDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(testDirectory, "..");
const siteRoot = resolve(repositoryRoot, "docs");

test("the sample catalog describes three songs and six surprise contours", () => {
  assert.equal(SONGS.length, 3);
  assert.equal(CONTOURS.length, 6);
  assert.equal(VIDEO_COUNT, 39);
});

test("each song and contour maps to the expected three comparison videos", () => {
  assert.deepEqual(getSampleSources("81", "a"), {
    groundTruth: "static/media/81-truth.mp4",
    surpriseNet: "static/media/81-type1.mp4",
    weighted: "static/media/81-weight-type1.mp4",
  });

  assert.deepEqual(getSampleSources("4", "f"), {
    groundTruth: "static/media/4-truth.mp4",
    surpriseNet: "static/media/4-type6.mp4",
    weighted: "static/media/4-weight-type6.mp4",
  });
});

test("every catalog media path resolves to a checked-in asset", async () => {
  const referencedFiles = new Set();

  for (const song of SONGS) {
    for (const contour of CONTOURS) {
      const sources = getSampleSources(song.id, contour.id);
      referencedFiles.add(sources.groundTruth);
      referencedFiles.add(sources.surpriseNet);
      referencedFiles.add(sources.weighted);
      referencedFiles.add(contour.thumbnail);
    }
  }

  await Promise.all(
    [...referencedFiles].map((path) => access(resolve(siteRoot, path))),
  );
});

test("the page is dependency-free and loads the app as a module", async () => {
  const html = await readFile(resolve(siteRoot, "index.html"), "utf8");

  assert.doesNotMatch(html, /jquery|bootstrap/i);
  assert.match(html, /static\/js\/app\.mjs/);
  assert.equal(html.match(/rel="preload"/g)?.length, 2);
  assert.match(html, /href="static\/media\/hero-480\.avif"[\s\S]+media="\(max-width: 56rem\)"/);
  assert.match(html, /href="static\/media\/hero-960\.avif"[\s\S]+media="\(min-width: 56\.001rem\)"/);
  assert.match(html, /<picture>[\s\S]+media="\(max-width: 56rem\)"[\s\S]+type="image\/avif"[\s\S]+<img/);
});

test("the demo explains that its samples are precomputed and backend-free", async () => {
  const [html, readme] = await Promise.all([
    readFile(resolve(siteRoot, "index.html"), "utf8"),
    readFile(resolve(repositoryRoot, "README.md"), "utf8"),
  ]);

  assert.match(html, /precomputed/i);
  assert.match(readme, /does not require a backend/i);
});

test("every static asset referenced by the page exists", async () => {
  const html = await readFile(resolve(siteRoot, "index.html"), "utf8");
  const assetPaths = [...html.matchAll(/(?:href|poster|src)="([^"]+)"/g)]
    .map(([, path]) => path)
    .filter((path) => path.startsWith("static/"));

  await Promise.all(assetPaths.map((path) => access(resolve(siteRoot, path))));
});

test("selector cards use dedicated contour thumbnails", () => {
  for (const contour of CONTOURS) {
    assert.match(contour.thumbnail, /-thumb\.avif$/);
    assert.equal("image" in contour, false);
  }
});

test("the page requires an explicit user action before loading any video", async () => {
  const html = await readFile(resolve(siteRoot, "index.html"), "utf8");

  assert.equal(html.match(/preload="none"/g)?.length, 3);
  assert.equal(html.match(/data-load-player=/g)?.length, 3);
  assert.doesNotMatch(html, /<video\b[^>]*\scontrols(?:\s|>)/);
  assert.equal(html.match(/poster="static\/media\/hero-480\.avif"/g)?.length, 3);
  assert.doesNotMatch(html, /poster="static\/media\/hero\.jpg"/);
  assert.doesNotMatch(html, /static\/media\/Type-[a-f]\.png/);
  assert.doesNotMatch(html, /\.mp4/);
});

test("video letterboxing uses the light score surface instead of black", async () => {
  const css = await readFile(resolve(siteRoot, "static/css/style.css"), "utf8");

  assert.match(css, /--media-surface:\s*#fff/);
  assert.equal(css.match(/background:\s*var\(--media-surface\)/g)?.length, 2);
  assert.doesNotMatch(css, /#071013/);
});

test("a late canplay event does not overwrite an active playing state", async () => {
  const { getCanPlayStatus } = await import("../docs/static/js/player-state.mjs");

  assert.deepEqual(getCanPlayStatus(false), {
    message: "Playing.",
    state: "playing",
  });
  assert.deepEqual(getCanPlayStatus(true), {
    message: "Ready to play.",
    state: "ready",
  });
});

test("an unchanged source preserves the active player state", async () => {
  const { hasPlayerSourceChanged } = await import(
    "../docs/static/js/player-state.mjs"
  );

  assert.equal(
    hasPlayerSourceChanged(
      "static/media/81-truth.mp4",
      "static/media/81-truth.mp4",
    ),
    false,
  );
  assert.equal(
    hasPlayerSourceChanged(
      "static/media/81-type1.mp4",
      "static/media/81-type2.mp4",
    ),
    true,
  );
});

test("a stale playback generation cannot update a newer selection", async () => {
  const { isPlaybackGenerationCurrent } = await import(
    "../docs/static/js/player-state.mjs"
  );

  assert.equal(isPlaybackGenerationCurrent(2, 4), false);
  assert.equal(isPlaybackGenerationCurrent(5, 5), true);
});

test("a retry action keeps its visible label in the accessible name", async () => {
  const { getLoadButtonLabel } = await import(
    "../docs/static/js/player-state.mjs"
  );

  assert.equal(
    getLoadButtonLabel("Try loading again", "SurpriseNet", "Breakdown"),
    "Try loading again SurpriseNet for Breakdown",
  );
});
