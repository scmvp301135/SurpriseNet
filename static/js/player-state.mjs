export function getCanPlayStatus(isPaused) {
  return isPaused
    ? { message: "Ready to play.", state: "ready" }
    : { message: "Playing.", state: "playing" };
}

export function getLoadButtonLabel(action, playerLabel, songTitle) {
  return `${action} ${playerLabel} for ${songTitle}`;
}

export function hasPlayerSourceChanged(currentSource, nextSource) {
  return currentSource !== nextSource;
}

export function isPlaybackGenerationCurrent(requestGeneration, currentGeneration) {
  return requestGeneration === currentGeneration;
}
