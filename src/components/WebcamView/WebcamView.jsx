import { useEffect, useRef } from 'react';
import { createHandLandmarker, drawHandLandmarks } from '../../utils/handDetector';
import './WebcamView.css';

export default function WebcamView({ onHandUpdate, landmarksRef }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const lastStatusRef = useRef(false);
  const onHandUpdateRef = useRef(onHandUpdate);

  // Garde la référence du callback à jour sans relancer useEffect
  useEffect(() => {
    onHandUpdateRef.current = onHandUpdate;
  }, [onHandUpdate]);

  useEffect(() => {
    let isMounted = true;
    let stream = null;
    let landmarker = null;
    let animationFrameId = null;
    let lastVideoTime = -1;

    async function setupCameraAndMediaPipe() {
      try {
        const detector = await createHandLandmarker();
        if (!isMounted) return;
        landmarker = detector;

        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
          audio: false
        });

        if (!isMounted) {
          mediaStream.getTracks().forEach(track => track.stop());
          return;
        }

        stream = mediaStream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        detectFrame();
      } catch (err) {
        if (isMounted) {
          console.error("Erreur d'initialisation :", err);
        }
      }
    }

    function detectFrame() {
      if (!isMounted) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (
        video &&
        canvas &&
        landmarker &&
        video.readyState >= 2 &&
        video.videoWidth > 0
      ) {
        const ctx = canvas.getContext('2d');

        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;

          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const results = landmarker.detectForVideo(video, performance.now());
          const hasHand = !!(results.landmarks && results.landmarks.length > 0);

          if (hasHand) {
            drawHandLandmarks(ctx, results.landmarks[0]);
            if (landmarksRef) landmarksRef.current = results.landmarks[0];
          } else {
            if (landmarksRef) landmarksRef.current = null;
          }

          if (onHandUpdateRef.current && hasHand !== lastStatusRef.current) {
            lastStatusRef.current = hasHand;
            onHandUpdateRef.current(hasHand);
          }
        }
      }

      animationFrameId = requestAnimationFrame(detectFrame);
    }

    setupCameraAndMediaPipe();

    return () => {
      isMounted = false;
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [landmarksRef]);

  return (
    <div className="webcam-container">
      <video ref={videoRef} className="webcam-feed" autoPlay playsInline muted />
      <canvas ref={canvasRef} className="output-canvas" width={640} height={480} />
    </div>
  );
}