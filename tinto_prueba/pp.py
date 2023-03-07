from tinto import tinto
experimento="5"
src_data = "C:\\Users\\borja\\PycharmProjects\\TINTORERA\\Datasets\\iris.csv"
folder = "C:\\Users\\borja\\PycharmProjects\\TINTORERA\\imgs\\"+experimento
tintonera=tinto()
#tintonera.saveHyperparameters()

tintonera.loadHyperparameters()
tintonera.generateImages()