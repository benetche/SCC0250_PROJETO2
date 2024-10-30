# Projeto 2 - Disciplina SCC0250 - Computação Gráfica (2024/2)

## Sobre o projeto
A cena representa o sentimento de solidão, com uma casa isolada no meio do deserto. \
O vazio da cena traz uma sensação de solidão e isolamento, enquanto o cachorro solitário passa a sensação de abandono.

## Controles 
* WASD - Movimentos da câmera
* P - ativa/desativa o modo malha
* 1-7 - Seleciona o objeto
* R - ativa o modo de rotação
* T - ativa o modo de translação 
* Z/X - aumenta/diminui o tamanho do objeto selecionado
### Modo rotaçao:
- seta para esquerda: diminui o angulo de rotação
- seta para direita: aumenta o angulo de rotação
### Modo translação:
- seta para esquerda: diminui a translação no eixo X
- seta para direita: aumenta a translação no eixo X
- seta para cima: diminui a translação no eixo Z
- seta para baixo: aumenta a translação no eixo Z
- espaço: aumenta a translação no eixo Y
- ctrl: diminui a translação no eixo Y

## Aspectos do projeto
O projeto foi desenvolvido em Python, utilizando as bibliotecas **GLFW**, **PyOpenGL** e **NumPy**. \
O chão foi feito utilizando um quadrado e mapeando uma textura de areia de forma repetida por todo o quadrado, dando maior resolução e suavidade. \
O SkyBox foi feito a partir de uma esfera circunscrita ao plano do chão. A textura então foi mapeada para cubrir toda superfície interna da esfera. A imagem da textura é grande o sufiecente para causar uma perspectiva de mundo ao redor. \
Os demais objetos foram importados de sites e mapeados com suas devidas texturas.\

Para impedir a câmera de atravessar o chão, após calcular a próxima posição da câmera com o movimento pelas teclas WASD, o código checa se a posição é valida, de acordo com o Ground Level e o Raio da Circunferência da SkyBox.

