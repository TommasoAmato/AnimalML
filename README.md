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

Il dataset di immagini di animali può essere scaricato dal seguente link: [link al dataset](https://www.kaggle.com/datasets/andrewmvd/animal-faces). Il dataset contiene 3 categorie di animali: gatti, cani e animali selvatici, ed è diviso in due cartelle: train e val.

## Addestramento del modello

Il modello di rete neurale convoluzionale (CNN) è stato addestrato utilizzando un'architettura composta da 2 layer Conv2D, 2 layer MaxPooling2D, 1 layer Flatten e 2 layer Dense. Il modello utilizza l'ottimizzatore Adam, la funzione di loss categorical_crossentropy e l'accuracy come metrica.

Puoi addestrare il modello eseguendo il seguente comando:

```bash
python train.py --train_data_path /path/to/train_data --val_data_path /path/to/val_data --num_classes 3 --epochs 10 --batch_size 32 --output_model_path /path/to/output/model
```
