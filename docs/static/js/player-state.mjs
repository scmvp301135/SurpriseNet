export function getCanPlayStatus(isPaused) {
  return isPaused
    ? { message: "Ready to play.", state: "ready" }
    : { message: "Playing.", state: "playing" };
}

export function hasPlayerSourceChanged(currentSource, nextSource) {
  return currentSource !== nextSource;
}

export function isPlaybackGenerationCurrent(requestGeneration, currentGeneration) {
  return requestGeneration === currentGeneration;
}
