#include <iostream>
#include <vector>
#include <cmath>
#include <random>

    float sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

	float calculate_neuron(const std::vector<float>& inputs, const std::vector<float>& weights, int size) {
		float sum = 0.0f;
		for (int i = 0; i < size; i++) {
			sum += inputs[i] * weights[i];
		}
        return sigmoid(sum);
	}

    struct Layer {
        std::vector<std::vector<float>> weights;
        std::vector<float> errors;
        std::vector<float> results;
        std::vector<float> inputs;


        void init(int n_neurons, int n_inputs) {
            errors.resize(n_neurons);
            results.resize(n_neurons);
            weights.resize(n_neurons);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

            for (int i = 0; i < n_neurons; i++) {
                weights[i].resize(n_inputs);
                for (int j = 0; j < n_inputs; j++) {
                    weights[i][j] = dis(gen);
                }
            }
        }
        void calculate_layer(int num_neurons, int num_inputs) {
            for (int i = 0; i < num_neurons; i++) {
                results[i] = calculate_neuron(inputs, weights[i], num_inputs);
            }
        }

        void educate_layers(const std::vector<float>& results, std::vector<std::vector<float>>& weights, const std::vector<float>& inputs, const std::vector<float>& errors, float lr) {
            for (int i = 0; i < results.size(); i++) {
                float sigmoid_deriv = results[i] * (1 - results[i]);
                float gradient = errors[i] * sigmoid_deriv;
                for (int j = 0; j < weights[i].size(); j++) {
                    weights[i][j] -= lr * gradient * inputs[j];
                }
            }
        }
        void calculate_errors(std::vector<int> config, int current_layer_size, int next_layer_size, std::vector<float>next_layer_errors, std::vector<float>current_results, std::vector<std::vector<float>> weights) {
            for (int i = 0; i < current_layer_size; i++) {
                float sum_error = 0.0;
                for (int j = 0; j < next_layer_size; j++) {
                    sum_error += next_layer_errors[j] * weights[j][i];
                }

                float res = current_results[i];
                errors[i] = sum_error * (res * (1.0f - res));
            }
        }
    };

        struct NeuralNetwork {
            std::vector<Layer> layers;

            void count_the_layer(const std::vector<int>& config, int first_inputs) {
                layers.resize(config.size());
                layers[0].init(config[0], first_inputs);
                for (int i = 1; i < config.size(); i++ ){
                    layers[i].init(config[i], config[i-1]);

                }
            }
            void calculate_result(std::vector<int> config, int first_inputs) {
                layers[0].calculate_layer(config[0], first_inputs);
                for (int i = 1; i < layers.size(); i++) {
                    layers[i].inputs = layers[i - 1].results;
                    layers[i].calculate_layer(config[i], config[i - 1]);
                }
            }
            void calculate_all_layers_errors(std::vector<int> config, int target, float last_res) {
                float last = config.size() - 1;
                layers[last].errors[0] = 2 * (last_res - target) * (last_res * (1.0f - last_res));
                for (int i = last - 1; i > 0; i--) {
                    layers[i].calculate_errors(config, layers[i].results.size(), layers[i+1].results.size(), layers[i+1].errors, layers[i].results, layers[i+1].weights);
                }
            }
            void train(int target, std::vector<float> first_inputs, std::vector<int> config, float lr) {
                layers[0].inputs = first_inputs;
                calculate_result(config, first_inputs.size());
                float actual_res = layers.back().results[0];
                calculate_all_layers_errors(config, target, actual_res);
                for (int i = 0; i < layers.size(); i++) {
                    layers[i].educate_layers(layers[i].results, layers[i].weights, layers[i].inputs, layers[i].errors, lr);
                }
            }
        };

        int main() {
            std::vector<int> config = { 2, 5, 1 };
            NeuralNetwork nn;

            nn.count_the_layer(config, 2);

            std::vector<float> input = { 0.5f, 0.1f };
            int target = 1;
            float lr = 0.05f;

            for (int i = 0; i < 10000; i++) {
                nn.train(target, input, config, lr);

                if (i % 100 == 0) {
                    std::cout << "Epoch " << i << " Output: " << nn.layers.back().results[0] << std::endl;
                }
            }
        }