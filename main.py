from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Generate los numeros aleatorios.
        random.seed(1)
        
        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
        
    """La funcion sigmoidea, que describe una curva en forma de S.
     Pasamos la suma ponderada de las entradas a través de esta función para normalizarlos entre 0 y 1."""
    def _sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    """  La derivada de la función sigmoidea.
     Este es el gradiente de la curva sigmoidea. Indica cuánta confianza tenemos sobre el peso existente."""
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    """ Formamos la red neuronal a través de un proceso de prueba y error.
       Ajustando los pesos sinápticos cada vez."""
    def train(self, training_set_inputs, training_set_outputs, number_iterations):
        for iteration in xrange(number_iterations):
            
            output = self.think(training_set_inputs) 
            
            """Calcular el error (La diferencia entre la salida deseada
             y la salida prevista)."""
            error = training_set_outputs - output
            
            """Multiplique el error por la entrada y nuevamente por el gradiente de la curva Sigmoide.
             Esto significa que los pesos menos seguros se ajustan más.
             Esto significa que las entradas, que son cero, no causan cambios en los pesos."""
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            
            """Se ajusta los pesos"""
            self.synaptic_weights += adjustment
    
    """La red neuronal piensa"""        
    def think(self, inputs):
        """Una sola neurona piensa"""
        return self._sigmoid(dot(inputs, self.synaptic_weights))
    

if __name__ == "_main_":
        
        neural_network = NeuralNetwork()
        
        """El conjunto de entrenamiento.Tenemos 4 ejemplos, cada uno consta de 3 valores de entrada y 1 valor de salida."""
        training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
        training_set_output = array([[0,1,1,0]]).T
        
        """Se entrena la red neuronal"""
        neural_network.train(training_set_inputs, training_set_output, 10000)
        
        print "Nuevos pesos despues del entrenamiento"  
        
        print neural_network.synaptic_weights
        
        print "Considering new situation [1,0,0] -> ? "
        print neural_network.think(array([1,0,0])) 
        
        
        