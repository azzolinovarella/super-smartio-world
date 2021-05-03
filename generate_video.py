import os
import pickle
import sys
import retro  # pip install gym-retro
import neat  # pip install neat-python
import cv2  # pip install opencv-python
import fornecidos_pelo_professor.utils as ut
import fornecidos_pelo_professor.rominfo as ri


full_path = os.path.dirname(__file__) 
op_sys = os.name


# Variáveis referentes à arquivos (colocadas aqui por facilidade em muda-las)
config_file = full_path + '/config.txt'
# Para a população base esses são os arquivos:
checkpoints_dir = full_path + '/checkpoints'
best_genome_file = checkpoints_dir + '/best_genome.pkl'
video_name = checkpoints_dir + '/best_genome.mp4'
# Quando desejar criar uma nova, esses são os arquivos:
ng_checkpoints_dir = full_path + '/ng-checkpoints'
ng_best_genome_file = ng_checkpoints_dir + '/ng-best_genome.pkl'
ng_video_name = ng_checkpoints_dir + '/ng-best_genome.mp4'


def main() -> None:
    """Função principal que executa o módulo de gerar o video do jogo. Basicamente
    é responsável por verificar se é desejado gerar a partir do melhor agente 
    dentre os treinados pelo autor ou o melhor treinado a partir de "train new"
    (i.e., a nova geração).
    """
    # Carrega as configurações (salvas no arquivo the configuração)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    if len(sys.argv) == 1:
        mk_video(config, best_genome_file, video_name)

    elif len(sys.argv) == 2 and sys.argv[1] == 'new':
        mk_video(config, ng_best_genome_file, ng_video_name)

    else:
        print('Opção inválida! ')
    
    return None


def mk_video(config: neat.config.Config, genome_path: str, path_to_save : str,
             level: str = 'YoshiIsland2') -> None:
    """Gera um vídeo em formato '.mp4' para o agente selecionado.

    Args:
        config (neat.config.Config): Arquivo de texto contendo as configurações
        para a biblioteca NEAT-Python (obrigatório) como o número de inputs, outputs,
        funções de ativação, tamanho da população e etc.
        genome_path (str): Path para onde está salvo o agente desejado.
        path_to_save (str): Path onde será salvo o vídeo.
        level (str, optional): É a fase que se deseja jogar. É por default 'YoshiIsland2' 
        (fase em que o agente foi treinado).
    """
    if os.path.exists(genome_path):
        with open(genome_path, 'rb') as file:
            genome = pickle.load(file)
    else:
        print('Arquivo inválido! Não foi encontrado nenhum agente.')
        return None
    
    display = input('Deseja ver o Mário jogando a fase enquanto obtemos os frames? (S/N) ').upper()
    while display not in ['S', 'N']:
        display = input('Opção inválida!\nDeseja ver o Mário jogando a fase enquanto obtemos os frames? (S/N) ').upper()
    
    if display == 'S': display = True
    else: display = False


    env = retro.make(game='SuperMarioWorld-Snes', state=level, players=1)
    env.reset()
    
    net = neat.nn.FeedForwardNetwork.create(genome, config) # Criando a Rede Neural

    frame = 0  # Utilizado para calcular quantos frames já passaram e verificar se o Mário ficou preso em um "Loop"
    count = 0  # Utilizado para verificar se o Mario travou
    done = False  # Usado para indicar quando o jogo terminará
    img_array = []
    print('Obtendo frames do jogo (isso pode demorar um pouco)...')
    while not done:
        inputs, mario_x, mario_y = ri.getInputs(ri.getRam(env))  # Pegando o vetor de entrada e a posição em x do Mário            
        output = net.activate(inputs)  # Pegando a saída da RNA ao utilizarmos o input acima
        action = [0 if bit < 0.5 else 1 for bit in output]  # Transformando em binário para simbolizar os botões            
        ob, _, _, _ = env.step(action)  # Mudamos o ambiente ao executar a ação
        if display: env.render()
        img_array.append(ob)  # Transforma de BGR para RBG 
        _, new_mario_x, _ = ri.getInputs(ri.getRam(env))

        # Para verificar possíveis problemas:
        ram = env.get_ram()
    
        # Indica que o Mário morreu
        if ram[0x0071] == 9: done = True

        # Verificar se o Mario travou
        if new_mario_x == mario_x: count += 1
        else: count = 0
        if count > 100: done = True

        # Sinaliza que o Mário ficou preso em um loop (e.g., indo repetidamente da esquerda para direita)
        marks = [v for v in range(500, 5500, 500)]  # Marcos úteis para verificar
        for mark in marks:
            if new_mario_x < mark < frame:  done = True 

        # Indica que o mário acabou a fase
        if ram[0x13D9] == 2: done = True

        while ram[0x1426] != 0:  # Mário abriu uma caixa de mensagem 
            ob, _, _, _, = env.step(ut.dec2bin(1))  # Botão que vai fazer ele fechar o balão de fala
            if display: env.render()
            img_array.append(ob) 
            ob, _, _, _, = env.step(ut.dec2bin(0))  # Não faz nada (Como se ele estivesse apertando e soltando o botão)
            if display: env.render()
            img_array.append(ob)   # Usado para selecionar se é desejado ver ou não a tela do Mário
            _, new_mario_x, _ = ri.getInputs(ri.getRam(env))
            ram = env.get_ram()
    
    # Resolução: y, x e cores   
    size_y, size_x, size_c = env.observation_space.shape
    new_size_x, new_size_y = 3 * size_x, 3 * size_y 
    
    if display: env.render(close=True)
    del env 

    # Reprocessando a imagem - aumentando qualidade e convertendo para RGB:
    img_array = [cv2.cvtColor(cv2.resize(img, (new_size_x, new_size_y)), cv2.COLOR_BGR2RGB) for img in img_array]

    out = cv2.VideoWriter(path_to_save, cv2.VideoWriter_fourcc(*'MP4V'), 45, (new_size_x, new_size_y))
    count = 0
    loading_bar = [' ' for i in range(0, 50)]  # Barra de loading utilizada para mostrar quanto do vídeo falta para ser feito
    # Gerando o vídeo
    for img in img_array: 
        out.write(img)

        if op_sys == 'nt':  # Windows
            os.system('cls')
        elif op_sys == 'posix':  # Linux/MacOS
            os.system('clear')

        percentage = round(count/len(img_array), 4)
        print(f'Criando vídeo: {(percentage * 100):.2f}%')
        loading_bar[round(percentage * 49)] =  '\u25A0'  # Unicode para quadrado preto (para preencher a barra)
        print(f'[{" ".join(loading_bar)}]')
        count += 1
    
    out.release() 

    if op_sys == 'nt':  # Windows
        os.system('cls')
    elif op_sys == 'posix':  # Linux/MacOS
        os.system('clear')

    print(f'Video salvo em: {path_to_save}')
    print(f'[{" ".join(loading_bar)}]')
    
    op = input('\nDeseja abrir o video agora? (S/N) ').upper()
    while op not in ['S', 'N']:
        op = input('Opção Inválida!\nDeseja abrir o video agora? (S/N) ').upper()
    
    if op == 'S':
        if op_sys == 'nt':  # Windows
            os.system(f'start {path_to_save}')
        elif op_sys == 'posix':
            os.system(f'xdg-open {path_to_save}')

    return None

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:  # Excluir o video
        print('\nCancelando operação...')
