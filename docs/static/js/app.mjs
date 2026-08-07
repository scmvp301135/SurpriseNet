import { CONTOURS, SONGS, getSampleSources } from "./catalog.mjs";
import {
  getCanPlayStatus,
  hasPlayerSourceChanged,
  isPlaybackGenerationCurrent,
} from "./player-state.mjs";

const songSelector = document.querySelector("#song-selector");
const contourSelector = document.querySelector("#contour-selector");
const selectionContext = document.querySelector("#selection-context");
const selectionTitle = document.querySelector("#selection-title");
const selectionDescription = document.querySelector("#selection-description");
const contourImage = document.querySelector("#current-contour-image");

const players = [
  {
    role: "groundTruth",
    label: "Ground truth",
    generation: 0,
    video: document.querySelector("#ground-truth-video"),
    status: document.querySelector("#ground-truth-status"),
    loadButton: document.querySelector('[data-load-player="groundTruth"]'),
  },
  {
    role: "surpriseNet",
    label: "SurpriseNet",
    generation: 0,
    video: document.querySelector("#surprisenet-video"),
    status: document.querySelector("#surprisenet-status"),
    loadButton: document.querySelector('[data-load-player="surpriseNet"]'),
  },
  {
    role: "weighted",
    label: "Weighted SurpriseNet",
    generation: 0,
    video: document.querySelector("#weighted-video"),
    status: document.querySelector("#weighted-status"),
    loadButton: document.querySelector('[data-load-player="weighted"]'),
  },
];

const initialParameters = new URLSearchParams(window.location.search);
const requestedSong = initialParameters.get("song");
const requestedContour = initialParameters.get("contour");

const state = {
  songId: SONGS.some(({ id }) => id === requestedSong) ? requestedSong : SONGS[0].id,
  contourId: CONTOURS.some(({ id }) => id === requestedContour)
    ? requestedContour
    : CONTOURS[0].id,
};

function createSongButton(song) {
  const button = document.createElement("button");
  const title = document.createElement("span");
  const detail = document.createElement("span");

  button.type = "button";
  button.className = "song-option";
  button.dataset.songId = song.id;
  button.setAttribute("aria-pressed", "false");

  title.className = "song-option-title";
  title.textContent = song.title;
  detail.className = "song-option-detail";
  detail.textContent = `${song.artist}, ${song.passage}`;

  button.append(title, detail);
  button.addEventListener("click", () => selectSong(song.id));
  return button;
}

function createContourButton(contour) {
  const button = document.createElement("button");
  const image = document.createElement("img");
  const label = document.createElement("span");

  button.type = "button";
  button.className = "contour-option";
  button.dataset.contourId = contour.id;
  button.setAttribute("aria-pressed", "false");
  button.setAttribute("aria-label", `${contour.name}. ${contour.description}`);

  image.src = contour.thumbnail;
  image.width = 240;
  image.height = 197;
  image.alt = "";
  image.loading = "lazy";
  image.decoding = "async";

  label.textContent = contour.name;
  button.append(image, label);
  button.addEventListener("click", () => selectContour(contour.id));
  return button;
}

function selectSong(songId) {
  if (state.songId === songId) return;
  state.songId = songId;
  renderSelection();
}

function selectContour(contourId) {
  if (state.contourId === contourId) return;
  state.contourId = contourId;
  renderSelection();
}

function setPlayerSource(player, source, song, contour) {
  const { video, status, loadButton, role, label } = player;
  const sourceChanged = hasPlayerSourceChanged(video.dataset.source, source);
  const ariaLabel =
    role === "groundTruth"
      ? `${label} for ${song.title}, ${song.passage}`
      : `${label} for ${song.title}, ${song.passage}, using the ${contour.name} contour`;

  video.poster = role === "groundTruth" ? "static/media/hero-480.avif" : contour.image;
  video.setAttribute("aria-label", ariaLabel);
  loadButton.setAttribute("aria-label", `Load and play ${label} for ${song.title}`);

  if (!sourceChanged) return;

  player.generation += 1;
  video.pause();
  video.removeAttribute("src");
  video.dataset.source = source;
  video.load();
  loadButton.hidden = false;
  loadButton.textContent = "Load and play";
  status.textContent = "Loads only when requested.";
  status.dataset.state = "idle";
}

async function loadPlayer(player) {
  const { video, loadButton } = player;
  const requestedSource = video.dataset.source;
  const requestGeneration = ++player.generation;

  if (!video.getAttribute("src")) {
    video.src = requestedSource;
    video.load();
  } else if (video.error) {
    video.load();
  }

  loadButton.hidden = true;
  video.focus();
  setStatus(player, "Preparing this sample.", "loading");

  try {
    await video.play();
  } catch {
    if (!isPlaybackGenerationCurrent(requestGeneration, player.generation)) return;

    loadButton.hidden = false;
    loadButton.textContent = "Try loading again";
    loadButton.focus();
    setStatus(player, "Playback did not start. Try again.", "error");
  }
}

function renderSelection() {
  const song = SONGS.find(({ id }) => id === state.songId);
  const contour = CONTOURS.find(({ id }) => id === state.contourId);
  const sources = getSampleSources(song.id, contour.id);

  for (const button of songSelector.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.songId === song.id));
  }

  for (const button of contourSelector.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.contourId === contour.id));
  }

  selectionContext.textContent = `${song.artist} / ${song.passage}`;
  selectionTitle.textContent = `${song.title}: ${contour.name}`;
  selectionDescription.textContent = contour.description;
  contourImage.src = contour.image;
  contourImage.alt = `${contour.name} surprise contour. ${contour.description}`;

  for (const player of players) {
    setPlayerSource(player, sources[player.role], song, contour);
  }

  const nextUrl = new URL(window.location.href);
  nextUrl.searchParams.set("song", song.id);
  nextUrl.searchParams.set("contour", contour.id);
  window.history.replaceState({ ...state }, "", nextUrl);
}

function moveSelectorFocus(event) {
  const directionalKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"];
  if (!directionalKeys.includes(event.key)) return;

  const buttons = [...event.currentTarget.querySelectorAll("button")];
  const currentIndex = buttons.indexOf(document.activeElement);
  if (currentIndex === -1) return;

  event.preventDefault();
  const direction = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1;
  const nextIndex = (currentIndex + direction + buttons.length) % buttons.length;
  buttons[nextIndex].focus();
  buttons[nextIndex].click();
}

function setStatus(player, message, stateName) {
  player.status.textContent = message;
  player.status.dataset.state = stateName;
}

for (const song of SONGS) {
  songSelector.append(createSongButton(song));
}

for (const contour of CONTOURS) {
  contourSelector.append(createContourButton(contour));
}

songSelector.addEventListener("keydown", moveSelectorFocus);
contourSelector.addEventListener("keydown", moveSelectorFocus);

for (const player of players) {
  player.loadButton.addEventListener("click", () => loadPlayer(player));

  player.video.addEventListener("loadstart", () => {
    setStatus(player, "Preparing this sample.", "loading");
  });

  player.video.addEventListener("canplay", () => {
    const nextStatus = getCanPlayStatus(player.video.paused);
    setStatus(player, nextStatus.message, nextStatus.state);
  });

  player.video.addEventListener("waiting", () => {
    setStatus(player, "Buffering audio and score.", "loading");
  });

  player.video.addEventListener("play", () => {
    for (const otherPlayer of players) {
      if (otherPlayer.video !== player.video) otherPlayer.video.pause();
    }
    setStatus(player, "Playing.", "playing");
  });

  player.video.addEventListener("pause", () => {
    if (!player.video.ended && player.video.currentTime > 0) {
      setStatus(player, "Paused.", "paused");
    }
  });

  player.video.addEventListener("ended", () => {
    setStatus(player, "Playback finished.", "ended");
  });

  player.video.addEventListener("error", () => {
    player.loadButton.hidden = false;
    player.loadButton.textContent = "Try loading again";
    setStatus(player, "This sample could not be loaded. Try again.", "error");
  });
}

renderSelection();
