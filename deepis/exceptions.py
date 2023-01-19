class WeightDimensionError(Exception):
    '''Raised when the weight dimension supplied to the neural net trainer
    does not match with the output dimension
    '''
    pass


class InputDimensionError(Exception):
    '''Raised when the input dimension supplied
    does not match with the neural net input dimension
    '''
    pass
