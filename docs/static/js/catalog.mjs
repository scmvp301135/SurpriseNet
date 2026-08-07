export const SONGS = Object.freeze([
  Object.freeze({
    id: "81",
    artist: "Joe Dassin",
    title: "Les Champs-Élysées",
    passage: "Verse",
  }),
  Object.freeze({
    id: "5",
    artist: "Jack Johnson",
    title: "Breakdown",
    passage: "Verse",
  }),
  Object.freeze({
    id: "4",
    artist: "Jack Johnson",
    title: "Breakdown",
    passage: "Chorus",
  }),
]);

export const CONTOURS = Object.freeze([
  Object.freeze({
    id: "a",
    index: 1,
    name: "Rising",
    description: "Surprise increases over time.",
    thumbnail: "static/media/Type-a-thumb.avif",
  }),
  Object.freeze({
    id: "b",
    index: 2,
    name: "Falling",
    description: "Surprise decreases over time.",
    thumbnail: "static/media/Type-b-thumb.avif",
  }),
  Object.freeze({
    id: "c",
    index: 3,
    name: "Low steady",
    description: "Surprise stays low.",
    thumbnail: "static/media/Type-c-thumb.avif",
  }),
  Object.freeze({
    id: "d",
    index: 4,
    name: "High steady",
    description: "Surprise stays high.",
    thumbnail: "static/media/Type-d-thumb.avif",
  }),
  Object.freeze({
    id: "e",
    index: 5,
    name: "Arch",
    description: "Surprise rises, then falls.",
    thumbnail: "static/media/Type-e-thumb.avif",
  }),
  Object.freeze({
    id: "f",
    index: 6,
    name: "Valley",
    description: "Surprise falls, then rises.",
    thumbnail: "static/media/Type-f-thumb.avif",
  }),
]);

export const VIDEO_COUNT = SONGS.length + SONGS.length * CONTOURS.length * 2;

export function getSampleSources(songId, contourId) {
  const song = SONGS.find(({ id }) => id === songId);
  const contour = CONTOURS.find(({ id }) => id === contourId);

  if (!song) {
    throw new RangeError(`Unknown song id: ${songId}`);
  }

  if (!contour) {
    throw new RangeError(`Unknown contour id: ${contourId}`);
  }

  return {
    groundTruth: `static/media/${song.id}-truth.mp4`,
    surpriseNet: `static/media/${song.id}-type${contour.index}.mp4`,
    weighted: `static/media/${song.id}-weight-type${contour.index}.mp4`,
  };
}
