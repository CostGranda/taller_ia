import array
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

def func_eval(individuo):
    # Sumatoria de 1s en el individuo.
    return sum(individuo),

# Clase que hereda de Fitness (Mide la calidad de una solución, Se le envian los valores iniciales en tupla(1.0) )
creator.create("AptitudMax", base.Fitness, weights=(1.0,))
creator.create("Individuo", array.array, typecode='b', fitness=creator.AptitudMax)

# Inicializa la "caja" de herramientas
toolbox = base.Toolbox()

# Generador de los atributos (0 y 1)
toolbox.register("attr_bool", random.randint, 0, 1)

# Inicializadores

# En este caso queremos que el mejor individuo contenga 20 uno (1)s
toolbox.register("individuo", tools.initRepeat, creator.Individuo, toolbox.attr_bool, 30)
# La población se rellena de listas de individuos
toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)

# Registra el operador de evaluación (nuestra función)
toolbox.register("evaluate", func_eval)

# Registra el operador de cruze (dos puntos)
toolbox.register("mate", tools.cxTwoPoint)

# Resgistra el operador de muación, con la probabilidad de que cada uno sea convertido
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# Operador para seleccionar los individuos para la crianza, tournsize es el numero de individuos participando
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    # Semilla para la aleatoriedad
    random.seed(64)

    # Se indica el tamaño de la población
    poblacion = toolbox.poblacion(n=300)
    # Contiene el mejor individuo que haya vivido en la población durante la evolución
    hof = tools.HallOfFame(1)


    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Promedio", numpy.mean)
    #stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Algoritmo por defecto que toma los valores de
    # Probabilidad de cruze (cxpb), Probabilidad de mutación (mutpb), Numero de Generaciones (ngen)
    # Las stadisticas que queremos mostrar (stats)
    # Retorna una tupla con el log, y la población final
    poblacion, log = algorithms.eaSimple(poblacion, toolbox, cxpb=0.5, mutpb=0.2, ngen=5,
                                   stats=stats, halloffame=hof, verbose=False)
    print(log)
    print("\n==== Fin de la evolución")
    # Selecciona El mejor individuo
    mejores_ind = tools.selBest(poblacion, 1)[0]
    print('\nMejor Individuo:\n', mejores_ind)
    print('\nNúmero de Unos:', sum(mejores_ind))
    #return poblacion, log, hof


if __name__ == "__main__":
    main()
