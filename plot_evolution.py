import os
import sys
import neat  # pip install neat-python
import matplotlib  # pip install matplotlib
import matplotlib.pyplot as plt  


full_path = os.path.dirname(__file__)  # -> Pega o path até aqui
op_sys = os.name

# Para a população base temos os arquivos:
checkpoints_dir = full_path + '/checkpoints'
checkpoints_prefix = 'neat-checkpoint-'
file_name = checkpoints_dir + '/evolution.png'
# Quando desejar criar uma nova, temos os arquivos:
ng_checkpoints_dir = full_path + '/ng-checkpoints'
ng_checkpoints_prefix = 'ng-neat-checkpoint-'
ng_file_name = ng_checkpoints_dir + '/ng-evolution.png'


def main() -> None:
    """Função principal que executa o módulo de gerar gráfico. Basicamente
    é responsável por verificar se é desejado plotar o gráfico para as 
    gerações treinadas pelo autor ou àquelas treinadas pelo "train new"
    (i.e., a nova geração).
    """
    if len(sys.argv) == 1:
        generate_graph(checkpoints_dir, checkpoints_prefix, file_name, True)
    
    elif len(sys.argv) == 2 and sys.argv[1] == 'new':
        generate_graph(ng_checkpoints_dir, ng_checkpoints_prefix, ng_file_name, True)
    
    else:
        print('Opção inválida!')

    return None


def generate_graph(checkpoint_dir: str, checkpoints_prefix: str, file_name: str,
                   display=True) -> None:
    """Gera um gráfico de Fitness x Geração, mostrando a Neuroevolução.

    Args:
        ccheckpoint_dir (str): Diretório onde estão salvos os checkpoints.
        checkpoints_prefix (str): Prefixo dos checkpoints.
        file_name (str): Nome do arquivo onde o gráfico (em formato png)
        será salvo.
        display (bool, optional): Booleano que indica se é desejado abrir uma
        tela mostrando o gráfico ou não. É por default True.
    """
    all_checkpoints = list(filter(lambda s: s.startswith(checkpoints_prefix), os.listdir(checkpoint_dir)))
    if len(all_checkpoints) != 0:  # Se houver pelo menos um checkpoint salvo        
        print('Atualizando gráfico...')
        all_checkpoints.sort(key=lambda s: int(s.split('-')[-1]))
        all_checkpoints = [checkpoint_dir + '/' + file for file in all_checkpoints]
        generations = []
        fitnesses = []

        for pop_file in all_checkpoints:
            pop = neat.Checkpointer.restore_checkpoint(pop_file)
            generations.append(int(pop.generation))
            fitnesses.append(max([0 if i.fitness is None else i.fitness for i in list(pop.population.values())]))

        try:
            if op_sys == 'posix' and display:
                matplotlib.use('TkAgg')  # sudo apt-get install python3-tk    
        except ModuleNotFoundError:
            display = False
            print("Impossível abrir gráfico, instale o tkinter por 'sudo apt-get install python3-tk'")


        # Abrir figura:
        plt.figure()
        plt.title('Fitness x Geração')
        plt.grid()
        # Eixo x:
        plt.xlabel('Geração')
        plt.xticks([v for v in range(min(generations), max(generations) + 5, 5 if len(generations) > 10 else 1)])
        # Eixo y:
        plt.ylabel('Fitness') 
        # Para plotar
        plt.plot(generations, fitnesses, marker='*')    
        plt.savefig(file_name)
        if display: plt.show()
        
        return None

    else:
        print('Não há nenhum checkpoint salvo!')
        
        return None


if __name__ == '__main__':
    try:
        main()
    
    except KeyboardInterrupt:
        if op_sys == 'nt':  # Windows
            os.system('cls')
        elif op_sys == 'posix':  # Linux/MacOS
            os.system('clear')

        print('\nCancelando geração de gráfico...\n') 
        