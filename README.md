# API de composition d'affiche — Jeunesse Africaine en Action

Ce backend remplit la partie §4.2 du cahier des charges qu'on ne peut pas faire
de façon fiable directement dans le navigateur : **suppression d'arrière-plan
par IA** (modèle U²-Net, via la librairie `rembg`), en plus de la détection de
visage et du recadrage intelligent.

## Ce que fait l'API

`POST /api/compose` reçoit une photo, et renvoie directement l'affiche
personnalisée finale en PNG HD — prête à télécharger ou partager.

1. Détection du visage principal (OpenCV Haar cascade — rapide, aucun modèle
   externe à télécharger, fonctionne hors-ligne).
2. Suppression d'arrière-plan (U²-Net via `rembg`) — le sujet est détouré
   proprement, comme s'il était réellement "dans" l'affiche.
3. Recadrage et redimensionnement automatique pour remplir exactement le
   cadre du visuel (mêmes coordonnées que la version navigateur : x=741,
   y=169, 473×652 px sur l'affiche 1254×1254).
4. Composition finale et renvoi de l'image — **rien n'est jamais écrit sur
   disque** : tout se passe en mémoire et est libéré à la fin de la requête
   (conforme à l'exigence RGPD du §4.3 — pas besoin du délai de 24h évoqué
   dans le cahier des charges, la suppression est immédiate).

## Lancer en local

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Puis tester :
```bash
curl -F "photo=@/chemin/vers/une_photo.jpg" http://localhost:8000/api/compose -o resultat.png
```

## Déployer en production

Point important : le modèle U²-Net pèse environ 170 Mo, ce qui **dépasse la
limite d'AWS Lambda / Google Cloud Functions** évoquée dans le cahier des
charges (250 Mo compressés, mais serré une fois les dépendances ajoutées).
Options recommandées, plus simples pour ce cas précis :

- **Render.com** ou **Railway** : déploiement direct depuis ce Dockerfile,
  gratuit ou très bon marché pour un trafic de forum ponctuel.
- **Google Cloud Run** : supporte les conteneurs Docker sans limite de
  taille de modèle, scale à zéro entre les pics de trafic (adapté à un
  événement avec pics d'usage autour des dates du forum).
- Lambda/Cloud Functions restent possibles avec un stockage du modèle sur
  un volume monté (EFS pour Lambda) si vous tenez à cette option.

Avec Docker :
```bash
docker build -t jeunesse-africaine-api .
docker run -p 8000:8000 jeunesse-africaine-api
```

## Brancher le frontend

Dans le fichier `jeunesse-africaine-affiche.html` déjà livré, il suffit
d'ajouter un appel à cette API à la place (ou en complément) du recadrage
côté navigateur :

```javascript
async function composeViaAPI(file) {
  const formData = new FormData();
  formData.append('photo', file);
  const response = await fetch('https://VOTRE-API-DEPLOYEE.example.com/api/compose', {
    method: 'POST',
    body: formData
  });
  if (!response.ok) throw new Error('Échec du traitement');
  const blob = await response.blob();
  return URL.createObjectURL(blob); // à afficher directement dans une <img> ou sur le <canvas>
}
```

Je recommande de garder les deux chemins disponibles : l'API pour la
meilleure qualité (arrière-plan supprimé), et le recadrage navigateur déjà
livré comme repli automatique si l'API est indisponible ou trop lente
(connexion faible, forte affluence le jour du forum).

## Limites connues

- Le détecteur Haar cascade est robuste mais moins précis que RetinaFace/MTCNN
  sur des visages de profil ou mal éclairés. Pour une précision supérieure,
  remplacer `get_face_cascade()` par un modèle MTCNN (librairie `mtcnn` ou
  `facenet-pytorch`), au prix d'un temps de traitement plus long.
- Le endpoint traite une requête à la fois de façon synchrone ; pour un fort
  trafic simultané le jour du forum, prévoir plusieurs instances (Cloud Run
  scale automatiquement) plutôt que d'augmenter les workers d'un seul
  conteneur (le modèle IA est gourmand en mémoire).
