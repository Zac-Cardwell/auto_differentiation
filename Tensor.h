#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <random>
#include <chrono>

// Function to generate a random number within a specified range
double random_number(double mean, double stddev);

struct GradInfo;

class Tensor{
    private:
        std::vector<float> data;    // data
        std::vector<float> grad;    // the current gradient
        std::vector<int> stride;    // Strides for indexing 
        bool requires_grad;         // Flag to compute gradients
        GradInfo* grad_info = nullptr;    // pointer to the gradient info of the parent operations

        //Because the data gets flattend this lets us still navigate it like a multi-dim 
        void calculateStride(); 

        // Flattens a multi-dim vector into a single-dim vector 
        template<typename M>
        void flatten(M& values, int depth=0);
        void flatten(std::vector<float> V1, int depth);

        void init_grad();

        int get_index(std::vector<int> indices);

    public:
        std::vector<int> shape; // Shape of Tensor

        // init tensor with vector
        template<typename M>
        Tensor(const M& values, bool requires_grad = true);
        // init blank tensor
        Tensor(bool requires_grad = true);
        // init tensor with given shape and fill it with initValue
        Tensor(const std::vector<int>& new_shape, float initValue=0, bool requires_grad = true);

        // Fill tensor with 0s
        void zeros(const std::vector<int>& shape);
        // Fill tensor with random valuels from a normal dist.
        void random(const std::vector<int>& shape, double mean=0.0, double stddev=1.0);

        // Get shape of tensor
        const int num_elements() const;

        bool check_gradinfo();

        // Access element at indices
        float& operator[](const std::vector<int>& indices);
        float& operator()(const int& index);
        const std::vector<float>& getData() const;

        // Common operations to overload
        template<typename M>
        void operator=(const M& values);
        Tensor operator+(Tensor& T1);
        Tensor operator-(Tensor& T1);
        Tensor operator*(Tensor& T1);
        Tensor operator/(Tensor& T1);
    
        // linear alg. operations
        Tensor matmul(Tensor& T2, bool t = false);
        void matmul_der(Tensor* grad_result);
        Tensor T();

        //get the current gradients
        std::vector<float> get_grad() const;

        //Clears gradients
        void clear_grad();

        //calculates gradients
        void backwards();

        void set_grad(int x = 1);
        void set_grad(std::vector<float> x);
};

struct GradInfo {
    Tensor* tensor;
    std::string operation;
    std::vector<Tensor*> inputs;

    GradInfo(Tensor* tensor, std::string operation, std::vector<Tensor*> inputs)
        : tensor(tensor), operation(operation), inputs(inputs) {}
};

#endif // TENSOR_H