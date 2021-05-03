import os
import sys
import pickle
import datetime as dt
import utils as ut
import retro  # pip install gym-retro
import neat  # pip install neat-python
import plot_evolution as plte


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
# Numero de gerações máximo e se é desejado mostrar ou não a janela do Mário:
num_max = 10  
display = True  


def main() -> None:
    """Função principal que executa o módulo de treino. Basicamente
    é responsável por verificar se é desejado treinar a partir das 
    gerações treinadas pelo autor ou uma nova, indicada pelo parâmetro
    "new" inserido junto a linha de comando.
    """
    global p  # Usaremos para recebermos a geração dentro de 'eval_genomes'

    # Carrega as configurações da biblioteca NEAT-Python (salvas no arquivo the configuração)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Caso não seja fornecida nenhuma variável, iremos continuar a população do último checkpoint
    if len(sys.argv) == 1: 
        # Gera uma lista com todos os arquivos neste diretório que tem o prefixo do checkpoint
        checkpoint_prefix = checkpoints_files.split('/')[-1]
        all_checkpoints = list(filter(lambda s: s.startswith(checkpoint_prefix), os.listdir(checkpoints_dir)))
        checkpoints_path = checkpoints_files
    
    elif len(sys.argv) == 2 and sys.argv[1] == 'new':
        if not os.path.exists(ng_checkpoints_dir):
            os.makedirs(ng_checkpoints_dir)
        
        checkpoint_list = [c for c in os.listdir(ng_checkpoints_dir) if c != ng_best_genome_file.split('/')[-1]]
        # Para escolher se a opção é recomeçar do zero ou treinar uma nova (caso haja check point): 
        if len(checkpoint_list) != 0:
            ans = input('Insira a opção desejada:\n(1) Continuar o útlimo treinamento\n(2) Iniciar um novo (fazer isso apagará toda a nova população - população base treinada pelo aluno não será alterada).\n(0) Sair\nInsira 1, 2 ou 0: ')
            while ans not in ['1', '2', '0']:
                ans = input('Insira a opção desejada:\n(1) Continuar o útlimo treinamento\n(2) Iniciar um novo (fazer isso apagará toda a nova população - população base treinada pelo aluno não será alterada).\n(0) Sair\nInsira 1, 2 ou 0: ')
            if ans == '0':
                sys.exit()
            elif ans == '2':
                # Removemos todos os arquivos ali salvos
                for file in os.listdir(ng_checkpoints_dir):
                    os.remove(ng_checkpoints_dir + '/' + file)
            
        checkpoint_prefix = ng_checkpoints_files.split('/')[-1]
        all_checkpoints = list(filter(lambda s: s.startswith(checkpoint_prefix), os.listdir(ng_checkpoints_dir)))            
        checkpoints_path = ng_checkpoints_files
        
    else:
        print('Opção inválida!')
        
        return None

    # Se houver pelo menos um checkpoint salvo e se quiser restaurar o ultimo checkpoint
    if len(all_checkpoints) != 0:  
        last_number = 0
        for c in all_checkpoints:
            number = int(c.split('-')[-1])  # Recebe o número de cada checkpoint 
            if number > last_number:
                last_number = number
        last_checkpoint = checkpoints_path + str(last_number)  # Gera a caminho até o último checkpoint
        p = neat.Checkpointer.restore_checkpoint(last_checkpoint)

    else:  # Se não houver nenhum checkpoint, inicia uma população do zero
        p = neat.Population(config)

    # Imprime algumas estatisticas na tela (como Fitness médio, desvpad, etc.)  
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.Checkpointer(1, filename_prefix=checkpoints_path))  # Salva 'checkpoints'

    # Executa a função 'eval_genomes', só finalizando quando gerar um total de 'num_max' gerações
    p.run(eval_genomes, num_max)

    return None


def eval_genomes(genomes: list, config: neat.config.Config) -> None:
    """O objetivo dessa função é selecionar os melhores indivíduos em 
    uma população atribuindo a cada um deles um Fitness para indicar o
    quão bom são.

    Args:
        genomes (list): Uma lista contendo todos os indivíduos da população (que
        são da classe 'neat.genome.DefaultGenome').
        config (neat.config.Config): Arquivo de texto contendo as configurações
        para a biblioteca NEAT-Python (obrigatório) como o número de inputs, outputs,
        funções de ativação, tamanho da população e etc.
    """
    global is_new_better  # Usado para verificar se foi encontrado um novo melhor indivíduo durante o treinamento

    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)

    print('\n################### INICIO ###################\n\n')
    # Fazendo uma iteração sobre todos os indivíduos da população para atribuir a cada um o Fitness correspondente
    for genome_id, genome in genomes:
        env.reset() 
        net = neat.nn.FeedForwardNetwork.create(genome, config)  # Criando a Rede Neural
        
        penalty = 0  # Penalidade dada ao Mário quando ele "enfrenta" cara a cara um inimigo
        frame = 0    # Utilizado para calcular quantos frames já passaram e verificar se o Mário ficou preso em um "Loop"
        count = 0  # Utilizado para verificar se o Mário travou no eixo x (dx = 0 -> c += 1) 
        t = 0  # Usado para medirmos o tempo gasto em cada fase
        done = False  # Usado para indicar quando a validação terminará
        print(f'# Individuo ID {genome_id} | População ID {p.generation} #')
        while not done:
            inputs, mario_x, mario_y = ri.getInputs(ri.getRam(env))  # Pegando o vetor de entrada e a posição em x do Mário            
            output = net.activate(inputs)  # Pegando a saída da RNA ao utilizarmos o input acima
            action = [0 if bit < 0.5 else 1 for bit in output]  # Transformando em binário para simbolizar os botões            
            env.step(action)  # Mudamos o ambiente ao executar a ação 
            if display: env.render()  # Usado para selecionar se é desejado ver ou não a tela do Mário         
            frame += 1  # Um frame a mais! 
            # Pega o novo vetor de entrada e a nova posição em x do Mário (usados para verificar se o agente ficou preso)
            new_inputs, new_mario_x, new_mario_y = ri.getInputs(ri.getRam(env))

            # Recebendo a RAM do jogo:           
            ram = env.get_ram()    
                        
            # Indica que o Mário morreu
            if ram[0x0071] == 9:  done = True 
            
            # Verificar se o Mário travou
            if new_mario_x == mario_x: count += 1
            else: count = 0
            if count > 100: done = True  # Se ficou parado em x quando count bate 100, encerramos a validação

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
            
            # Mário abriu uma caixa de mensagem (Yoshi ou Bloco de informação)
            while ram[0x1426] != 0:  
                env.step(ut.dec2bin(1))  # Botão que vai fazer ele fechar o balão de fala
                if display: env.render()  # Usado para selecionar se é desejado ver ou não a tela do Mário  
                env.step(ut.dec2bin(0))  # Não faz nada (Como se ele estivesse apertando e soltando o botão)
                if display: env.render()  # Usado para selecionar se é desejado ver ou não a tela do Mário
                new_inputs, new_mario_x, new_mario_y = ri.getInputs(ri.getRam(env))
                ram = env.get_ram()

            # Pegando o tempo in game
            if ram[0x0F30] == 0:  t += 1

        # Atualiza o fitness do indivíduo
        genome.fitness = float(fitness)    
        
        # Selecionamos o arquivo com base no argv
        if len(sys.argv) == 1:
            bgf = best_genome_file
        elif len(sys.argv) == 2 and sys.argv[1] == 'new':
            bgf = ng_best_genome_file
        
        # Se for o melhor, salvamos ele
        if os.path.exists(bgf):
            with open(bgf, 'rb') as file:
                best_genome = pickle.load(file)
            if genome.fitness > best_genome.fitness:
                best_genome  = genome
                is_new_better = True
                with open(bgf, 'wb') as file:
                    pickle.dump(genome, file)

        else:  # Se não existir o arquivo de melhor genome, criamos e salvamos o genome atual
            best_genome = genome
            is_new_better = True
            with open(bgf, 'wb') as file:
                    pickle.dump(genome, file)
                
        # Imprime as características desse indivíduo
        print(f'==================== INFO ====================\n'
              f'-> Geração: {p.generation}\n'
              f'-> ID do Indivíduo: {genome_id}\n'
              f'-> Posição final: ({new_mario_x}, {new_mario_y})\n'
              f'-> Penalidade total: {penalty:.2f}\n'
              f'-> Fitness: {genome.fitness:.2f}\n'
              f'-> Tempo gasto (no jogo): {t}s\n'
              f'-> É o melhor: {genome.fitness == best_genome.fitness}\n'
              f'==============================================\n\n')

    print('\n##################### FIM ####################\n\n')
    
    env.render(close=True) 
    del env

    return None    


if __name__ == '__main__':
    t0 = dt.datetime.now()
    is_new_better = False  # Se for gerado um melhor, mudaremos esse valor!
    try: 
        main()
        
    except KeyboardInterrupt:
        print('\nFinalizando treinamento...')

    tf = dt.datetime.now()

    # Para atualizar o gráfico: 
    if len(sys.argv) == 1:
        plte.generate_graph(plte.checkpoints_dir, plte.checkpoints_prefix, plte.file_name, False)
    elif len(sys.argv) == 2 and sys.argv[1] == 'new':
        plte.generate_graph(plte.ng_checkpoints_dir, plte.ng_checkpoints_prefix, plte.ng_file_name, False)

    if op_sys == 'nt':  # Windows
        os.system('cls')
    elif op_sys == 'posix':  # Linux/MacOS
        os.system('clear')

    if len(sys.argv) == 1 and os.path.exists(best_genome_file):
        with open(best_genome_file, 'rb') as file:
            best_genome = pickle.load(file)
        
    elif len(sys.argv) == 2 and sys.argv[1] == 'new' and os.path.exists(ng_best_genome_file):
        with open(ng_best_genome_file, 'rb') as file:
            best_genome = pickle.load(file)

    else:
        print('Não foi possível encontrar nenhum melhor indivíduo ou a opção inserida de inicio é inválida!')
        sys.exit()

    print(f'==================== INFO ====================\n'
          f'-> Melhor fitness: {best_genome.fitness:.2f}\n'
          f'-> Foi melhorado nesse treino: {is_new_better}\n'
          f'-> Tempo gasto no treinamento: {(tf - t0).seconds // 60} min e {(tf - t0).seconds % 60} s\n'
          f'==============================================\n\n')
    