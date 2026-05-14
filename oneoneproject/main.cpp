#include <iostream>
#include <vector>
#include <cmath>
#include <random>

struct Layer {
	std::vector<std::vector<float>> weights;
	std::vector<float>results;
	std::vector<float>errors;
	std::vector<float>bias;
	
	void init_layer_size(int num_neurons, int num_inputs){
		bias.resize(num_neurons, 0.01f);
		results.resize(num_neurons);
		errors.resize(num_neurons);
		weights.resize(num_neurons);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis((-0.05f, 0.05f));

		for (int i = 0; i < num_neurons; i++) {
			weights[i].resize(num_inputs);
			for (int j = 0; j < num_inputs; j++) {
				weights[i][j] = dis(gen);
			}
		}
	}
	float relu_d(float x) {
		if (x > 0) return 1.0f;
		return 0.01f;
	}
	void calculate_all_neuron_in_layer(const std::vector<std::vector<float>>& weights, const std::vector<float>& inputs, std::vector<float>& results, int num_neuron, const std::vector<float>&bias) {
		for (int j = 0; j < num_neuron; j++) {
			float sum = 0.0f;
			for (int i = 0; i < weights[j].size(); i++) {
				sum += weights[j][i] * inputs[i];
			}
			sum += bias[j];
			results[j] = (sum > 0.0f) ? sum : 0.01f;
		}
	}
	void calculate_error_in_last_layer(float leaning_rate, std::vector<float>& errors, const std::vector<float>& targets, const std::vector<float>& results) {
		for (int i = 0; i < results.size(); i++) {
			errors[i] = 2 * (results[i] - targets[i]);
		}
	}
	void calculate_all_error_in_layer(float leaning_rate, std::vector<float>& errors, const std::vector<float>& errors_next, const std::vector<float>& results, const std::vector<std::vector<float>>& next_weights) {
		for (int i = 0; i < errors.size(); i++) {
			float sum = 0.0f;
			for (int j = 0; j < errors_next.size(); j++) {
				sum += errors_next[j] * next_weights[j][i];
			}
			errors[i] = sum * relu_d(results[i]);
		}
	}
	void educate_all_neurons_in_layer(float leaning_rate, const std::vector<float>& errors, const std::vector<float>& results, const std::vector<float>& inputs, std::vector<std::vector<float>>& weights, std::vector<float>& bias) {
		for (int i = 0; i < results.size(); i++) {
			float gradient = errors[i] * relu_d(results[i]);
			for (int j = 0; j < inputs.size(); j++) {
				weights[i][j] -= leaning_rate * gradient * inputs[j];
			}
			bias[i] -= leaning_rate * gradient;
		}
	}
};

struct Layers {
	std::vector<Layer>layers;

	void init_all_layers(const std::vector<int>& config, const std::vector<float>& first_inputs) {
		layers[0].init_layer_size(config[0], first_inputs.size());
		for (int i = 1; i < config.size(); i++) {
			layers[i].init_layer_size(config[i], config[i - 1]);
		}
	}
	void calculate_neuron_in_all_layer(const std::vector<int>& config, const std::vector<float>& first_inputs) {
		layers[0].calculate_all_neuron_in_layer(layers[0].weights, first_inputs, layers[0].results, layers[0].results.size(), layers[0].bias);
		for (int i = 1; i < config.size(); i++) {
			layers[i].calculate_all_neuron_in_layer(layers[i].weights, layers[i - 1].results, layers[i].results, layers[i].results.size(), layers[i].bias);
		}
	}
	void calculate_error_in_all_layer(float leaning_rate, const std::vector<int>& config, const std::vector<float>& targets) {
		int last = config.size() - 1;
		layers[last].calculate_error_in_last_layer(leaning_rate, layers[last].errors, targets, layers[last].results);
		for (int i = config.size() - 2; i >= 0; i--) {
			layers[i].calculate_all_error_in_layer(leaning_rate, layers[i].errors, layers[i+1].errors, layers[i].results, layers[i + 1].weights);
		}
	}
	void educate_neurons_in_all_layer(const std::vector<int>& config, float leaning_rate, const std::vector<float>& first_inputs) {
		layers[0].educate_all_neurons_in_layer(leaning_rate, layers[0].errors, layers[0].results, first_inputs, layers[0].weights, layers[0].bias);
		for (int i = 1; i < config.size(); i++) {
			layers[i].educate_all_neurons_in_layer(leaning_rate ,layers[i].errors, layers[i].results, layers[i-1].results, layers[i].weights, layers[i].bias);
		}
	}
	void init(const std::vector<int>& config, const std::vector<float>& first_inputs) {
		layers.resize(config.size());
		init_all_layers(config, first_inputs);
	}
	void train(const std::vector<int>& config, const std::vector<float>& first_inputs, const std::vector<float>& targets, float leaning_rate) {
		for (int i = 0; i < layers.size(); i++) {
			layers[i].results.assign(layers[i].results.size(), 0.0f);
			layers[i].errors.assign(layers[i].errors.size(), 0.0f);
		}
		calculate_neuron_in_all_layer(config, first_inputs);
		calculate_error_in_all_layer(leaning_rate, config, targets);
		educate_neurons_in_all_layer(config, leaning_rate, first_inputs);
	}
};

int main() {
	Layers net;
	std::vector<int> config = { 4, 2, 9, 2};
	std::vector<float> input = { 1.0f, 0.5f };
	std::vector<float> target = { 5.0f, 23.0f};
	net.init(config, input);

	for (int epoch = 0; epoch < 2000; epoch++) {
		net.train(config, input, target, 0.001f);

		if (epoch % 10 == 0) {
			std::cout << "epoch: " << epoch << " / ";

			const std::vector<float>& final_results = net.layers.back().results;
			for (size_t i = 0; i < final_results.size(); ++i) {
				std::cout << "result " << (i + 1) << ": " << final_results[i] << "    ";
			}

			std::cout << std::endl;
		}
	}
	return 0;
}