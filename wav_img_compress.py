"""
Parte 1: Leitura e Organização dos Dados
Nesta parte, vamos implementar uma classe que será responsável 
por ler o diretório de imagens e organizar os dados em um 
DataFrame do Pandas. A classe terá um método para carregar as
imagens do diretório e outro para salvar o DataFrame em um 
arquivo CSV.
"""

import cv2
import os
import pandas as pd

class ImageDataset:
    """
    A classe ImageDataset tem um construtor que recebe o caminho do diretório com as imagens. 
    O método load_images é responsável por encontrar todas as imagens no diretório, carregá-las
    e armazená-las em um DataFrame do Pandas. O método save_to_csv salva o DataFrame em um
    arquivo CSV para uso posterior.
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.image_files = None
        self.data = None
    
    def load_images(self):
        # Encontra todos os arquivos de imagem no diretório
        self.image_files = [os.path.join(self.dir_path, f) for f in os.listdir(self.dir_path) if f.endswith('.jpg')]
        
        # Carrega as imagens e armazena em um DataFrame do Pandas
        data = []
        for file in self.image_files:
            img = cv2.imread(file)
            data.append({'file_path': file, 'image': img})
        self.data = pd.DataFrame(data)
    
    def save_to_csv(self, file_path):
        # Salva o DataFrame em um arquivo CSV
        self.data.to_csv(file_path)

"""
Parte 2: Compressão e Redimensionamento
Nesta parte, vamos implementar as funções de compressão e redimensionamento 
utilizando a transformada wavelet e a interpolação. Essas funções serão 
implementadas em uma classe ImageProcessor.
"""

import pywt

class ImageProcessor:
    """
    A classe ImageProcessor tem um construtor vazio e dois métodos: compress e resize.
    O método compress aplica a transformada wavelet na imagem, calcula o limiar de compressão
    com base na porcentagem dos detalhes e zera os coeficientes abaixo desse limiar. 
    Em seguida, reconstrói a imagem a partir dos coeficientes comprimidos.
    O método resize redimensiona a imagem utilizando a interpolação bicúbica.
    """
    def __init__(self):
        self.wavelet = None
    
    def compress(self, img, wavelet_type='haar', level=1, threshold_percent=0.1):
        # Aplica a transformada wavelet na imagem
        coeffs = pywt.wavedec2(img, wavelet_type, level=level)
        
        # Calcula o limiar de compressão baseado na porcentagem dos detalhes
        threshold = threshold_percent * max([abs(c).max() for c in coeffs[1:]])
        
        # Zera os coeficientes abaixo do limiar
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            new_coeffs.append(pywt.threshold(coeffs[i], threshold))
        
        # Reconstrói a imagem a partir dos coeficientes comprimidos
        compressed_img = pywt.waverec2(new_coeffs, wavelet_type)
        return compressed_img.astype('uint8')
    
    def resize(self, img, resolution):
        # Redimensiona a imagem utilizando a interpolação bicúbica
        resized_img = cv2.resize(img, resolution, interpolation=cv2.INTER_CUBIC)
        return resized_img

"""
Parte 3: Validação Cruzada Automática
Nesta parte, vamos implementar a funcionalidade de validação cruzada automática 
para avaliar os parâmetros de compressão. Essa funcionalidade será implementada 
em uma classe ImageValidator.
"""

import numpy as np
from sklearn.model_selection import KFold

class ImageValidator:
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    
    def validate(self, wavelet_types, levels, threshold_percents, resolutions):
        # Separa os dados em treino e teste utilizando validação cruzada
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        kf_indices = kfold.split(self.dataset.data)
        
        # Validação cruzada para avaliar a compressão
        best_score = 0
        best_params = {}
        for wavelet_type in wavelet_types:
            for level in levels:
                for threshold_percent in threshold_percents:
                    for resolution in resolutions:
                        scores = []
                        for train_indices, test_indices in kf_indices:
                            train_data = self.dataset.data.iloc[train_indices]
                            test_data = self.dataset.data.iloc[test_indices]
                            
                            # Comprime as imagens de treino
                            train_images = train_data['image'].values
                            compressed_train_images = [self.processor.compress(img, wavelet_type, level, threshold_percent) for img in train_images]
                            
                            # Redimensiona as imagens de treino
                            resized_train_images = [self.processor.resize(img, resolution) for img in compressed_train_images]
                            resized_train_images = np.array(resized_train_images)
                            
                            # Comprime as imagens de teste
                            test_images = test_data['image'].values
                            compressed_test_images = [self.processor.compress(img, wavelet_type, level, threshold_percent) for img in test_images]
                            
                            # Redimensiona as imagens de teste
                            resized_test_images = [self.processor.resize(img, resolution) for img in compressed_test_images]
                            resized_test_images = np.array(resized_test_images)
                            
                            # Avalia a qualidade da compressão utilizando o índice SNR
                            snr = self.compute_snr(resized_test_images, test_images)
                            scores.append(snr)
                        
                        # Calcula a média do índice SNR para a combinação de parâmetros
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {'wavelet_type': wavelet_type, 'level': level, 'threshold_percent': threshold_percent, 'resolution': resolution}
        
        return best_params
    
    def compute_snr(self, x, y):
        # Calcula o índice SNR entre duas imagens
        mse = np.mean((x - y)**2)
        psnr = 10 * np.log10((255**2) / mse)
        return psnr
