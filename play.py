import os
import pickle
import time
import sys
import gzip
import retro  # pip install gym-retro
import neat  # pip install neat-python
import numpy as np  # pip install numpy
from colorama import Fore, init  # pip install colorama
import fornecidos_pelo_professor.utils as ut
import fornecidos_pelo_professor.rominfo as ri

init(autoreset=True)  # Iniciando colorama  --> vamos imprimir o que o mario ve em Cores
full_path = os.path.dirname(__file__)  # -> Pega o path até aqui
op_sys = os.name

# Variáveis referentes à arquivos (colocadas aqui para facilidade em muda-las)
config_file = full_path + '/config.txt'
# Para a população base esses são os arquivos:
checkpoints_dir = full_path + '/checkpoints'
checkpoints_files = checkpoints_dir + '/neat-checkpoint-'
best_genome_file = checkpoints_dir + '/best_genome.pkl'
# Quando desejar criar uma nova, esses são os arquivos:
ng_checkpoints_dir = full_path + '/ng-checkpoints'
ng_checkpoints_files = ng_checkpoints_dir + '/ng-neat-checkpoint-'
ng_best_genome_file = ng_checkpoints_dir + '/ng-best_genome.pkl'


def main() -> None:
    """Função principal que executa o módulo de jogar o jogo. Basicamente
    é responsável por verificar se é desejado jogar a partir do melhor agente 
    dentre os treinados pelo autor, o melhor treinado a partir de "train new"
    (i.e., a nova geração) ou todos aqueles treinados em um checkpoint (seja da
    geração treinada pelo autor ou pelo usuário).
    """
    # Carrega as configurações (salvas no arquivo the configuração)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Condicionais abaixo usados para escolher o conjunto de genomes a jogarem 
    if len(sys.argv) == 1:  # Padrão -> Joga só o melhor agente treinado por mim
        if os.path.exists(best_genome_file):
            with open(best_genome_file, 'rb') as file:
                genomes = [pickle.load(file)]
        else:
            print('Não foi encontrado o arquivo contendo o melhor agente. Treine a população a partir de:\n'
                  '"python -m train"')
            return None

    elif len(sys.argv) == 2: 
        if os.path.exists(checkpoints_dir):
            checkpoints = [c for c in os.listdir(checkpoints_dir) if c != best_genome_file.split('/')[-1]]
        else: 
            checkpoints = []
        
        if os.path.exists(ng_checkpoints_dir):
            ng_checkpoints = [c for c in os.listdir(ng_checkpoints_dir) if c != ng_best_genome_file.split('/')[-1]]
        else: 
            ng_checkpoints =[]

        if sys.argv[1] == 'new':  # Joga só o melhor agente treinado pela nova geração
            if os.path.exists(ng_best_genome_file):
                with open(ng_best_genome_file, 'rb') as file:
                    genomes = [pickle.load(file)]
            else:
                print('Não foi encontrado o arquivo contendo o melhor agente da nova população. Treine a nova população a partir de:\n'
                      '"python -m train new"')
                return None 
        
        elif sys.argv[1] in checkpoints:
            checkpoint = checkpoints_dir + '/' + sys.argv[1]
            with gzip.open(checkpoint, 'rb') as file:
                _, _, population, _, _ = pickle.load(file)
            genomes = [g for g in population.values()]


        elif sys.argv[1] in ng_checkpoints:
            ng_checkpoint = ng_checkpoints_dir + '/' + sys.argv[1]
            with gzip.open(ng_checkpoint, 'rb') as file:
                _, _, population, _, _ = pickle.load(file)
            genomes = [g for g in population.values()]
        
        else:
            print('Opção inválida ou checkpoint não encontrado! ')
            return None

    else:
        print('Opção inválida! ')
        return None
    
    play_game(genomes, config)

    return None


def play_game(genomes: list, config: neat.config.Config, 
              level: str = 'YoshiIsland2', display : bool = True) -> None:  
    """Faz com que um agente ou uma população fornecido(das) em um arquivo '.pkl'
    jogue a fase selecionada. 

    Args:
        genomes (list): Lista de individuos que jogarão a fase.
        config (neat.config.Config): Arquivo de texto contendo as configurações
        para a biblioteca NEAT-Python (obrigatório) como o número de inputs, outputs,
        funções de ativação, tamanho da população e etc.
        level (str, optional): É a fase que se deseja jogar. É por default 'YoshiIsland2' 
        (fase em que o agente foi treinado).
        display (bool, optional): Booleano que indica se é desejado (ou não) abrir a tela 
        mostrando o Mario jogando a fase. É por default True
    """     
    env = retro.make(game='SuperMarioWorld-Snes', state=level, players=1) 

    info = {}  # Usado para imprimir ao final o melhor individuo encontrado
    best_fitness = - np.inf
    for genome in genomes:
        env.reset()

        net = neat.nn.FeedForwardNetwork.create(genome, config) # Criando a Rede Neural

        penalty = 0  # Penalidade dada ao Mário quando ele "enfrenta" cara a cara um inimigo
        frame = 0  # Utilizado para calcular quantos frames já passaram e verificar se o Mário ficou preso em um "Loop"
        count = 0  # Utilizado para verificar se o Mario travou
        t = 0  # Usaremos para medir o tempo do mário
        done = False  # Usado para indicar quando o jogo terminará
        while not done:
            inputs, mario_x, mario_y = ri.getInputs(ri.getRam(env))  # Pegando o vetor de entrada e a posição em x do Mário            
            output = net.activate(inputs)  # Pegando a saída da RNA ao utilizarmos o input acima
            action = [0 if bit < 0.5 else 1 for bit in output]  # Transformando em binário para simbolizar os botões            
            env.step(action)  # Mudamos o ambiente ao executar a ação 
            if display: env.render()  # Usado para selecionar se é desejado ver ou não a tela do Mário            
            frame += 1  # Um frame a mais! 
             # Pega o novo vetor de entrada e a nova posição em x do Mário (usados para verificar se o agente ficou preso)
            new_inputs, new_mario_x, new_mario_y = ri.getInputs(ri.getRam(env))
                            
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

            # Posições 97 e 98 são logo a frente do Mário; Usaremos para desencorajar Mário a "enfrentar" os inimigos "cara a cara"
            if -1 in (new_inputs[97], new_inputs[98]):  penalty += 5   
            
            # Indica que o mário acabou a fase
            if ram[0x13D9] == 2:
                # Usaremos isso para inserir o fator tempo entre os melhores que terminaram a fase (decidir pelo mais rapido) 
                fitness = new_mario_x + 1000 * (390 - t)/390 
                done = True
            else:
                # Fitness é a posição x que o Mário conseguiu alcançar descontado de uma penalização
                fitness = new_mario_x - penalty

            while ram[0x1426] != 0:  # Mário abriu uma caixa de mensagem 
                env.step(ut.dec2bin(1))  # Botão que vai fazer ele fechar o balão de fala
                if display: env.render()  # Usado para selecionar se é desejado ver ou não a tela do Mário  
                env.step(ut.dec2bin(0))  # Não faz nada (Como se ele estivesse apertando e soltando o botão)
                if display: env.render()  # Usado para selecionar se é desejado ver ou não a tela do Mário
                new_inputs, new_mario_x, new_mario_y = ri.getInputs(ri.getRam(env))
                ram = env.get_ram()
            
            # Pegando o tempo in game
            if ram[0x0F30] == 0:  t += 1


            if op_sys == 'nt':  # Windows
                os.system('cls')
            elif op_sys == 'posix':  # Linux/MacOS
                os.system('clear')
            
            m_input = np.resize(inputs,(13,13)).astype(str) 
            m_input[7, 6] = f'{Fore.MAGENTA}\u25A0'
            m_input = np.char.replace(m_input, '-1', f'{Fore.GREEN}\u25A0')
            m_input = np.char.replace(m_input, '1', f'{Fore.BLUE}\u25A0')
            m_input = np.char.replace(m_input, '0', f'{Fore.WHITE}\u25A1')
            m_vision = '\n'.join([' '.join(line) for line in m_input])

            fitness = float(fitness)  # Para padronizar com train
            print(f'==================== INFO ====================\n'
                  f'-> Posição do Mário: ({new_mario_x}, {new_mario_y})\n'
                  f'-> Penalidade: {penalty:.2f}\n'
                  f'-> Fitness: {fitness:.2f}\n'
                  f'-> Tempo gasto (no jogo): {t}s\n'
                  f'==============================================\n\n'
                  f'Visão do Mário: \n{m_vision}')
        
        # Limpa a tela antes de imprimir a última informação 
        if op_sys == 'nt':  # Windows
            os.system('cls')
        elif op_sys == 'posix':  # Linux/MacOS
            os.system('clear')

        # Imprime ao chegar no final
        print(f'Informações finais deste agente: \n')
        print(f'==================== INFO ====================\n'
              f'-> Posição final do Mário: ({new_mario_x}, {new_mario_y})\n'
              f'-> Penalidade total: {penalty:.2f}\n'
              f'-> Fitness total obtido nessa fase: {fitness:.2f}\n'
              f'-> Tempo gasto (no jogo): {t}s\n'
              f'==============================================\n\n'
              f'Visão final do Mário: \n{m_vision}\n\n{Fore.RESET}Essa tela será fechada em 5s (Não encerre o programa)...')

        # Congela a tela do Mário por 5 segundos antes de fechar o jogo
        time.sleep(5)
        # Atualiza as informações para imprimirmos o melhor no final
        if fitness > best_fitness:
            best_fitness = fitness
            info["final_position"] = (new_mario_x, new_mario_y)
            info["penalty"] = penalty
            info["fitness"] = fitness
            info["time"] = t

    env.render(close=True)
    del env 

    if op_sys == 'nt':  # Windows
        os.system('cls')
    elif op_sys == 'posix':  # Linux/MacOS
        os.system('clear')

    print(f'Neste arquivo, o melhor indivíduo é caracterizado por:')
    print(f'==================== INFO ====================\n'
          f'-> Posição final do Mário: {info["final_position"]}\n'
          f'-> Penalidade total: {info["penalty"]:.2f}\n'
          f'-> Fitness total obtido nessa fase: {info["fitness"]:.2f}\n'
          f'-> Tempo gasto (no jogo): {info["time"]}s\n'
          f'==============================================\n\n')

    return None   


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        if op_sys == 'nt':  # Windows
            os.system('cls')
        elif op_sys == 'posix':  # Linux/MacOS
            os.system('clear')

        print('\nFinalizando jogo...\n')
        