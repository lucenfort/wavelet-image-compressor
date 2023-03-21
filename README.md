# Wavelet Image Compressor

O Wavelet Image Compressor é um programa em Python que permite a compressão de imagens utilizando algoritmos de transformada wavelet e redimensionamento. O objetivo do programa é reduzir o tamanho das imagens de forma eficiente, mantendo a qualidade e a informação necessária para sua utilização.

## Como usar

Para utilizar o programa, basta executar o arquivo `wavelet_image_compressor.py` e passar os argumentos necessários:

```bash
python wavelet_image_compressor.py --input /caminho/para/imagem.png --output /caminho/para/nova_imagem.png --wavelet haar --level 2 --threshold 0.1 --size 800 600
```

### Onde:

--input é o caminho para o arquivo de entrada (imagem a ser comprimida).
--output é o caminho para o arquivo de saída (imagem comprimida e redimensionada).
--wavelet é o tipo de wavelet utilizado (padrão: 'haar')
--level é o nível de decomposição da wavelet (padrão: 1)
--threshold é o valor mínimo para os coeficientes da wavelet após a decomposição (padrão: 0.1)
--size é o tamanho da imagem de saída (padrão: 800x600)

## Funcionamento

O programa utiliza o algoritmo de compressão JPEG 2000, que é baseado na transformada wavelet. A compressão JPEG 2000 é considerada uma das mais eficientes em termos de taxa de compressão e qualidade de imagem. Além disso, o programa utiliza um algoritmo de redimensionamento de imagem baseado em interpolação bilinear, que mantém a qualidade da imagem mesmo após o redimensionamento.

O programa oferece uma interface simples e fácil de usar, permitindo que o usuário especifique os parâmetros da compressão e do redimensionamento de forma intuitiva. Além disso, o programa é altamente customizável, permitindo que o usuário escolha diferentes tipos de wavelet, níveis de decomposição, limiares de compressão, entre outros parâmetros.
