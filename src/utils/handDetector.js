import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

let handLandmarker = null;

// Initialise l'instance de MediaPipe Hand Landmarker
export async function createHandLandmarker() {
  if (handLandmarker) return handLandmarker;

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: '/models/hand_landmarker.task',
      delegate: 'GPU' // Utilise l'accélération graphique locale
    },
    runningMode: 'VIDEO',
    numHands: 1
  });

  return handLandmarker;
}

// Fonction pour dessiner les points et connexions de la main sur le canvas
export function drawHandLandmarks(ctx, landmarks) {
  if (!landmarks || landmarks.length === 0) return;

  // Connexions entre les 21 points de la main
  const CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],        // Pouce
    [0, 5], [5, 6], [6, 7], [7, 8],        // Index
    [5, 9], [9, 10], [10, 11], [11, 12],   // Majeur
    [9, 13], [13, 14], [14, 15], [15, 16], // Annulaire
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20] // Auriculaire
  ];

  const width = ctx.canvas.width;
  const height = ctx.canvas.height;

  // 1. Tracer les lignes d'articulation
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 3;

  CONNECTIONS.forEach(([i, j]) => {
    const pt1 = landmarks[i];
    const pt2 = landmarks[j];

    ctx.beginPath();
    ctx.moveTo(pt1.x * width, pt1.y * height);
    ctx.lineTo(pt2.x * width, pt2.y * height);
    ctx.stroke();
  });

  // 2. Dessiner les points des repères (landmarks)
  landmarks.forEach((pt) => {
    ctx.beginPath();
    ctx.arc(pt.x * width, pt.y * height, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#22c55e';
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
  });
}