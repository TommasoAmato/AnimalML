# AnimalML

Questo progetto utilizza il _supervised learning_ per riconoscere le immagini di animali. Il modello di rete neurale convoluzionale (CNN) è stato addestrato utilizzando un dataset di immagini di gatti, cani e animali selvatici.

## Dipendenze

Per eseguire il progetto, è necessario installare le seguenti librerie Python:

- Tensorflow
- Keras
- NumPy
- Pandas
- Pillow
- Scikit-learn

Puoi installare le librerie utilizzando il seguente comando:

```bash
pip install tensorflow keras numpy pandas pillow scikit-learn
```
## Dataset

Il dataset di immagini di animali può essere scaricato dal seguente [link](https://www.kaggle.com/datasets/andrewmvd/animal-faces). Il dataset contiene 3 categorie di animali: gatti, cani e animali selvatici, ed è diviso in due cartelle: train e val.

## Addestramento del modello

Il modello di rete neurale convoluzionale (CNN) è stato addestrato utilizzando un'architettura composta da 2 layer Conv2D, 2 layer MaxPooling2D, 1 layer Flatten e 2 layer Dense. Il modello utilizza l'ottimizzatore Adam, la funzione di loss categorical_crossentropy e l'accuracy come metrica.

## Valutazione del modello

Dopo aver addestrato il modello, puoi valutare le sue prestazioni utilizzando i dati di test. Il file AnimalML.py valuta il modello sui dati di test e stampa l'accuracy del modello.

## Predizione su nuove immagini

Puoi utilizzare il modello addestrato per fare predizioni su nuove immagini cambiando il percorso all'interno del file AnimalML.py