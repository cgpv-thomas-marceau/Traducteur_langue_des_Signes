import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import './SignPredictor.css';

const STORAGE_KEY = 'sign_language_knn_dataset';
const FEATURE_SIZE = 63; // 21 points * 3 coordonnées (x, y, z)

function landmarksToTensor(landmarks) {
  //recupere la position du poignet
  const wrist = landmarks[0];
  let maxDist = 0;

  //soustrait la position du poignet a chaque point 
  // ce qui permet de comparer les donnee peut importe la position de la main dans l'ecran
  const relative = landmarks.map((pt) => {
    const dx = pt.x - wrist.x;
    const dy = pt.y - wrist.y;
    const dz = pt.z - wrist.z;
    const dist = Math.hypot(dx, dy, dz);
    if (dist > maxDist) maxDist = dist;
    return [dx, dy, dz];
  });

  const scale = maxDist || 1;
  //dx/scale divise toute les distance par le taill max de la main afin que peut importe la distance de la main tensorflow voit les memes chiffre
  //flatmap met les 21 tableaux de 3 dans 1 tableau de 63
  const flat = relative.flatMap(([dx, dy, dz]) => [dx / scale, dy / scale, dz / scale]);
  //converti le tableau javascript en tensor compris par la carte graphique / tensorflow
  return tf.tensor1d(flat);
}

//bloc de code generer par gemini
// Convertit le dataset KNN en objet JSON serialisable
function serializeDataset(classifier) {
  const dataset = classifier.getClassifierDataset();
  const exportObj = {};
  Object.keys(dataset).forEach((key) => {
    exportObj[key] = Array.from(dataset[key].dataSync());
  });
  return exportObj;
}

// Recharge un objet JSON dans le classifieur KNN et retourne le compteur d'exemples
function deserializeDataset(classifier, dataObj) {
  const tensorObj = {};
  const counts = { A: 0, B: 0, C: 0, L: 0, V: 0 };

  Object.keys(dataObj).forEach((label) => {
    const flatData = dataObj[label];
    const numSamples = flatData.length / FEATURE_SIZE;
    if (numSamples > 0) {
      tensorObj[label] = tf.tensor2d(flatData, [numSamples, FEATURE_SIZE]);
      counts[label] = numSamples;
    }
  });

  if (Object.keys(tensorObj).length > 0) {
    classifier.setClassifierDataset(tensorObj);
  }
  return counts;
}
//fin du bloc de code fait par gemini

export default function SignPredictor({ landmarksRef, isHandDetected }) {
  const [predictedSign, setPredictedSign] = useState('?');
  const [confidence, setConfidence] = useState(0);
  const [samplesCount, setSamplesCount] = useState({ A: 0, B: 0, C: 0, L: 0, V: 0 });
  const [isModelReady, setIsModelReady] = useState(false);

  //permet de garder les donner meme a si on actualise la page 
  // l idee vient de gemini
  const classifierRef = useRef(null);
  const fileInputRef = useRef(null);

  // Initialisation + chargement automatique depuis localStorage
  //la parti qui fait la sauvgarde des donnee d'aprentissage a ete fait par gemini
  useEffect(() => {
    let isMounted = true;
    async function initTF() {
      // attend que tensorflow soit pret
      await tf.ready();
      if (!isMounted) return;

      const classifier = knnClassifier.create();
      
      
      //verifie si un apprentissage a deja ete fait sur le navigateur. Si oui. le restaure
      const savedData = localStorage.getItem(STORAGE_KEY);

      if (savedData) {
        try {
          const parsed = JSON.parse(savedData);
          const restoredCounts = deserializeDataset(classifier, parsed);
          setSamplesCount(restoredCounts);
        } catch (err) {
          console.error('Erreur de restauration du modèle sauvegardé :', err);
        }
      }

      classifierRef.current = classifier;
      setIsModelReady(true);
    }
    initTF();
    return () => { isMounted = false; };
  }, []);

  // Boucle de prédiction en direct
  useEffect(() => {
    let isPredicting = false;

    // met un intervale de 150 millisecondes (6 a 7 fois par seconde) entre chaque analyse 
    // ce qui a l'aire instantanne pour l'oeil humain mais permet au processeur de souffler
    const intervalId = setInterval(async () => {
      //empeche de relancer une analyse si la precedente n'est pas terminer
      if (isPredicting) return;

      const landmarks = landmarksRef?.current;
      const classifier = classifierRef.current;

      //si aucune lettre n'est rentrer ou aucune main n'est devant la camera affiche 0% et ?
      if (!landmarks || !classifier || classifier.getNumClasses() === 0) {
        setPredictedSign((prev) => (prev !== '?' ? '?' : prev));
        setConfidence((prev) => (prev !== 0 ? 0 : prev));
        return;
      }

      isPredicting = true;
      try {
        const tensor = landmarksToTensor(landmarks);
        //compare les 63 valeur actuelle de la main a tout les exemple enregistree
        const result = await classifier.predictClass(tensor);
        //détruit manuellement le tenseur de la mémoire vidéo
        tensor.dispose();

        setPredictedSign(result.label);
        const score = Math.round((result.confidences[result.label] || 0) * 100);
        setConfidence(score);
      } finally {
        isPredicting = false;
      }
    }, 150);

    //coupe le minuteur si le composant est fermer
    return () => clearInterval(intervalId);
  }, [landmarksRef]);

  // 3. Collecter un exemple + sauvegarder automatiquement
  const handleCollectSample = (label) => {
    const landmarks = landmarksRef?.current;
    const classifier = classifierRef.current;
    if (!landmarks || !classifier) return;

    const tensor = landmarksToTensor(landmarks);
    classifier.addExample(tensor, label);
    tensor.dispose();

    // Sauvegarde automatique dans le navigateur
    //fait par gemini
    localStorage.setItem(STORAGE_KEY, JSON.stringify(serializeDataset(classifier)));

    setSamplesCount((prev) => ({
      ...prev,
      [label]: (prev[label] || 0) + 1,
    }));
  };
  //debut bloc genere par gemini
  //Exporter en fichier JSON
  const handleExportDataset = () => {
    const classifier = classifierRef.current;
    if (!classifier || classifier.getNumClasses() === 0) return;

    const exportObj = serializeDataset(classifier);
    const blob = new Blob([JSON.stringify(exportObj)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'modele-langue-des-signes.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  //Importer un fichier JSON existant
  const handleImportDataset = (event) => {
    const file = event.target.files?.[0];
    const classifier = classifierRef.current;
    if (!file || !classifier) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const parsed = JSON.parse(e.target.result);
        classifier.clearAllClasses();
        const restoredCounts = deserializeDataset(classifier, parsed);
        setSamplesCount(restoredCounts);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(parsed));
      } catch (err) {
        console.error('Fichier JSON invalide :', err);
      }
    };
    reader.readAsText(file);
    event.target.value = ''; // Permet de réimporter le même fichier si besoin
  };

  //Réinitialiser tout l'apprentissage
  const handleResetDataset = () => {
    const classifier = classifierRef.current;
    if (!classifier) return;
    classifier.clearAllClasses();
    localStorage.removeItem(STORAGE_KEY);
    setSamplesCount({ A: 0, B: 0, C: 0, L: 0, V: 0 });
    setPredictedSign('?');
    setConfidence(0);
  };

  return (
    <div className="predictor-panel">
      <div className="card">
        <h2>Prédiction en direct</h2>
        <div className="prediction-box">
          <div className="sign-display">{predictedSign}</div>
          <div className="confidence-box">
            <div>Confiance : <strong>{confidence}%</strong></div>
            <div className="progress-bar-bg">
              <div className="progress-bar-fill" style={{ width: `${confidence}%` }} />
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Collecte des données (Dataset)</h2>
        <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
          {isModelReady
            ? 'Tes échantillons sont sauvegardés automatiquement dans le navigateur.'
            : 'Chargement de TensorFlow.js...'}
        </p>

        <div className="dataset-buttons">
          {['A', 'B', 'C', 'L', 'V'].map((label) => (
            <button
              key={label}
              className="btn btn-primary"
              disabled={!isHandDetected || !isModelReady}
              onClick={() => handleCollectSample(label)}
            >
              {label} ({samplesCount[label] || 0})
            </button>
          ))}
        </div>

        <div className="dataset-actions">
          <button className="btn btn-export" onClick={handleExportDataset}>
            Exporter (JSON)
          </button>
          <button className="btn" onClick={() => fileInputRef.current?.click()}>
            Importer (JSON)
          </button>
          <button className="btn btn-danger" onClick={handleResetDataset}>
            Réinitialiser
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={handleImportDataset}
        />
      </div>
    </div>
  );
}
//fin bloc generer par gemini