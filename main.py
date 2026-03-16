# ============================================================
#  Mini AI Model Trainer Framework
#  School of Artificial Intelligence (SAI) — OOP Project
# ============================================================

from abc import ABC, abstractmethod   # Required for Abstraction


# ============================================================
# CLASS 1: ModelConfig
# OOP Concepts: Instance Attributes, Magic Method (__repr__)
# ============================================================
class ModelConfig:
    """Stores hyperparameter settings for a model."""

    def __init__(self, model_name, learning_rate=0.01, epochs=10):
        # OOP: Instance Attributes — unique to each ModelConfig object
        self.model_name    = model_name
        self.learning_rate = learning_rate
        self.epochs        = epochs

    def __repr__(self):
        # OOP: Magic Method — defines how object prints as string
        return f"[Config] {self.model_name} | lr={self.learning_rate} | epochs={self.epochs}"


# ============================================================
# CLASS 2: BaseModel  (Abstract Base Class)
# OOP Concepts: Abstraction, Class Attribute, Composition
# ============================================================
class BaseModel(ABC):
    """Abstract base class — defines the interface all models must follow."""

    # OOP: Class Attribute — shared across ALL instances of BaseModel
    model_count = 0

    def __init__(self, config: ModelConfig):
        # OOP: Composition — BaseModel OWNS a ModelConfig (has-a relationship)
        # ModelConfig is created inside the model and lives with it
        self.config = config

        # OOP: Class Attribute update — increment every time a model is created
        BaseModel.model_count += 1

    # OOP: Abstraction — forces ALL child classes to implement these methods
    @abstractmethod
    def train(self, data):
        """Train the model on given data."""
        pass

    @abstractmethod
    def evaluate(self, data):
        """Evaluate the model on given data."""
        pass

    def summary(self) -> str:
        # OOP: Instance Method — available to all subclasses
        return f"Model: {self.config.model_name} | Epochs: {self.config.epochs} | LR: {self.config.learning_rate}"


# ============================================================
# CLASS 3: LinearRegressionModel
# OOP Concepts: Single Inheritance, Method Overriding, super()
# ============================================================
class LinearRegressionModel(BaseModel):
    """Concrete Linear Regression model — inherits from BaseModel."""

    def __init__(self, learning_rate=0.01, epochs=10):
        # OOP: Composition — create a ModelConfig object to pass to parent
        config = ModelConfig("LinearRegression", learning_rate, epochs)

        # OOP: super() — calls BaseModel.__init__ to properly initialize parent
        super().__init__(config)

    # OOP: Method Overriding — provides LR-specific implementation of train()
    def train(self, data):
        # OOP: Instance Method
        print(f"LinearRegression: Training on {len(data)} samples "
              f"for {self.config.epochs} epochs (lr={self.config.learning_rate})")

    # OOP: Method Overriding — provides LR-specific implementation of evaluate()
    def evaluate(self, data):
        # OOP: Instance Method — simulated MSE metric
        print("LinearRegression: Evaluation MSE = 0.042")


# ============================================================
# CLASS 4: NeuralNetworkModel
# OOP Concepts: Single Inheritance, Method Overriding, super(),
#               Extra Instance Attribute (layers)
# ============================================================
class NeuralNetworkModel(BaseModel):
    """Concrete Neural Network model — inherits from BaseModel."""

    def __init__(self, layers=None, learning_rate=0.001, epochs=20):
        # OOP: Instance Attribute — extra attribute specific to NeuralNetworkModel
        self.layers = layers if layers is not None else [64, 32, 1]

        # OOP: Composition — create config and pass to parent
        config = ModelConfig("NeuralNetwork", learning_rate, epochs)

        # OOP: super() — calls BaseModel.__init__
        super().__init__(config)

    # OOP: Method Overriding — NN-specific train() (different output than LR)
    def train(self, data):
        # OOP: Instance Method
        print(f"NeuralNetwork {self.layers}: Training on {len(data)} samples "
              f"for {self.config.epochs} epochs (lr={self.config.learning_rate})")

    # OOP: Method Overriding — NN-specific evaluate() (Accuracy instead of MSE)
    def evaluate(self, data):
        # OOP: Instance Method — simulated Accuracy metric
        print("NeuralNetwork: Evaluation Accuracy = 91.5%")


# ============================================================
# CLASS 5: DataLoader
# OOP Concepts: Instance Attributes, Instance Method
# Independent class — not in the inheritance chain
# ============================================================
class DataLoader:
    """Holds a dataset and provides access to it."""

    def __init__(self, dataset):
        # OOP: Instance Attribute
        self.dataset = dataset

    def get_data(self):
        # OOP: Instance Method — returns the stored dataset
        return self.dataset


# ============================================================
# CLASS 6: Trainer
# OOP Concepts: Aggregation, Polymorphism
# ============================================================
class Trainer:
    """Orchestrates the full training pipeline: load → train → evaluate."""

    def __init__(self, model: BaseModel, loader: DataLoader):
        # OOP: Aggregation — Trainer BORROWS DataLoader (uses-a relationship)
        # DataLoader is created OUTSIDE and passed in — Trainer doesn't own it
        self.model  = model
        self.loader = loader

    def run(self):
        # OOP: Polymorphism — run() works with ANY subclass of BaseModel
        # It calls train() and evaluate() without knowing which model it is
        print(f"\n--- Training {self.model.config.model_name} ---")
        data = self.loader.get_data()       # get data from DataLoader
        self.model.train(data)              # polymorphic call
        self.model.evaluate(data)           # polymorphic call


# ============================================================
# MAIN — Demonstrate the full framework
# ============================================================
if __name__ == "__main__":

    # Create two model instances
    lr_model = LinearRegressionModel(learning_rate=0.01, epochs=10)
    nn_model = NeuralNetworkModel(layers=[64, 32, 1], learning_rate=0.001, epochs=20)

    # OOP: Magic Method (__repr__) — prints config in formatted form
    print(lr_model.config)
    print(nn_model.config)

    # OOP: Class Attribute — accessed on the class itself
    print(f"Models created: {BaseModel.model_count}")

    # Create DataLoader — independent of models (Aggregation)
    loader = DataLoader([1.2, 2.4, 3.1, 4.8, 5.5])

    # OOP: Polymorphism — Trainer.run() works for BOTH models without modification
    Trainer(lr_model, loader).run()
    Trainer(nn_model, loader).run()