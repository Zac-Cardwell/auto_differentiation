#include "Tensor.h"
#include <iostream>

// Function to generate a random number within a specified range
double random_number(double mean, double stddev) {
    static std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(generator);
}

// Initialize the gradient vector with zeros
void Tensor::init_grad() {
    grad.resize(num_elements(), 0.0);
}

// Calculate the stride for each dimension of the tensor
void Tensor::calculateStride() {
    stride.resize(shape.size());
    int s = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        stride[i] = s;
        s *= shape[i];
    }
}

// Template function to flatten nested vectors and determine the shape
template<typename M>
void Tensor::flatten(M& values, int depth) {
    if (depth >= shape.size()) {
        shape.resize(depth + 1);
    }
    shape[depth] = values.size();
    for (const auto& v : values) {
        flatten(v, depth + 1);
    }
}

// Specialization of flatten function for 1D vector
void Tensor::flatten(std::vector<float> V1, int depth) {
    if (depth >= shape.size()) {
        shape.resize(depth + 1);
    }
    shape[depth] = V1.size();
    for (float i : V1) {
        data.push_back(i);
    }
}

// Get the linear index from multi-dimensional indices
int Tensor::get_index(std::vector<int> indices) {
    int index = 0;
    for (int i = 0; i < shape.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Tensor index out of range");
        }
        index += indices[i] * stride[i];
    }
    return index;
}

// Constructor to initialize tensor from nested vectors
template<typename M>
Tensor::Tensor(const M& values, bool requires_grad): requires_grad(requires_grad) {
    flatten(values, 0);  // Start flattening from top-level
    calculateStride();
    if (requires_grad) {
        init_grad();
    }
}

// Default constructor
Tensor::Tensor(bool requires_grad): requires_grad(requires_grad) { 
    if (requires_grad) {
        init_grad();
    }
}

// Constructor to initialize tensor with a specific shape and initial value
Tensor::Tensor(const std::vector<int>& shape, float initValue, bool requires_grad)
    : shape(shape), requires_grad(requires_grad) {
    for (int i = 0; i < num_elements(); i++) {
        data.push_back(initValue);
    }
    calculateStride();
    if (requires_grad) {
        init_grad();
    }
}



// Check if gradient information is available
bool Tensor::check_gradinfo() {
    return grad_info != nullptr;
}

// Initialize tensor with random values
void Tensor::random(const std::vector<int>& new_shape, double mean, double stddev) {
    shape = new_shape;
    data.clear();
    grad.clear();
    for (int i = 0; i < num_elements(); i++) {
        data.push_back(random_number(mean, stddev));
    }
    calculateStride();
    if (requires_grad) {
        init_grad();
    }
}

// Initialize tensor with zeros
void Tensor::zeros(const std::vector<int>& shape) {
    this->shape = shape;
    data.assign(num_elements(), 0.0f);
    calculateStride();
    if (requires_grad) {
        init_grad();
    }
}

// Get the total number of elements in the tensor
const int Tensor::num_elements() const {
    int size = 1;
    for (int i : shape) {
        size *= i;
    }
    return size;
}

// Overload operator[] to access elements using multi-dimensional indices
float& Tensor::operator[](const std::vector<int>& indices) {
    if (indices.size() != shape.size()) {
        throw std::out_of_range("Incorrect number of indices");
    }

    int index = 0;
    for (int i = 0; i < shape.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Tensor index out of range");
        }
        index += indices[i] * stride[i];
    }
    return data[index];
}

// Overload operator() to access elements using a linear index
float& Tensor::operator()(const int& index) {
    return data[index];
}

// Get the underlying data of the tensor
const std::vector<float>& Tensor::getData() const {
    return data;
}

// Assign new values to the tensor
template<typename M>
void Tensor::operator=(const M& values) {
    data.clear();
    shape.clear();
    flatten(values, 0);  // Start flattening from top-level
    calculateStride();
}


// Overload operator+ for tensor addition
Tensor Tensor::operator+(Tensor& T1) {
    if (shape != T1.shape) {
        throw std::invalid_argument("Shapes of the tensors do not match for addition.");
    }

    Tensor result(shape, 0.0, requires_grad || T1.requires_grad);
    for (int i = 0; i < num_elements(); ++i) {
        result.data[i] = data[i] + T1.data[i];
    }

    if (requires_grad || T1.requires_grad) {
        result.requires_grad = true;
        result.grad_info = new GradInfo(&result, "+", std::vector<Tensor*>{this, &T1});
    }

    return result;
}


// Overload operator- for tensor subtraction
Tensor Tensor::operator-(Tensor& T1) {
    if (shape != T1.shape) {
        throw std::invalid_argument("Shapes of the tensors do not match for subtraction.");
    }

    Tensor result(shape, 0.0, requires_grad || T1.requires_grad);
    for (int i = 0; i < num_elements(); ++i) {
        result.data[i] = data[i] - T1.data[i];
    }

    if (requires_grad || T1.requires_grad) {
        result.grad_info = new GradInfo(&result, "-", std::vector<Tensor*>{this, &T1});
    }

    return result;
}


// Overload operator* for element-wise tensor multiplication
Tensor Tensor::operator*(Tensor& T1) {
    if (shape != T1.shape) {
        throw std::invalid_argument("Shapes of the tensors do not match for multiplication.");
    }

    Tensor result(shape, 0.0, requires_grad || T1.requires_grad);
    for (int i = 0; i < num_elements(); ++i) {
        result.data[i] = data[i] * T1.data[i];
    }

    if (requires_grad || T1.requires_grad) {
        result.grad_info = new GradInfo(&result, "*", std::vector<Tensor*>{this, &T1});
    }

    return result;
}


// Overload operator/ for element-wise tensor division
Tensor Tensor::operator/(Tensor& T1) {
    if (shape != T1.shape) {
        throw std::invalid_argument("Shapes of the tensors do not match for division.");
    }

    Tensor result(shape, 0.0, requires_grad || T1.requires_grad);
    for (int i = 0; i < num_elements(); ++i) {
        result.data[i] = data[i] / T1.data[i];
    }

    if (requires_grad || T1.requires_grad) {
        result.grad_info = new GradInfo(&result, "/", std::vector<Tensor*>{this, &T1});
    }

    return result;
}

//// Perform backpropagation to compute gradients
void Tensor::backwards(){
    if (!requires_grad || !grad_info) {
            return;
    }else if (grad_info->operation == "+") {
        for (int i = 0; i < num_elements(); ++i) {
            grad_info->inputs[0]->grad[i] += grad[i];
            grad_info->inputs[1]->grad[i] += grad[i];
        }
        grad_info->inputs[0]->backwards();
        grad_info->inputs[1]->backwards();
    }else if (grad_info->operation == "-") {
    for (int i = 0; i < num_elements(); ++i) {
        grad_info->inputs[0]->grad[i] += grad[i];
        grad_info->inputs[1]->grad[i] -= grad[i];
    }
    grad_info->inputs[0]->backwards();
    grad_info->inputs[1]->backwards();
    }else if (grad_info->operation == "*") {
        for (int i = 0; i < num_elements(); ++i) {
            grad_info->inputs[0]->grad[i] += grad[i] * grad_info->inputs[1]->data[i];
            grad_info->inputs[1]->grad[i] += grad[i] * grad_info->inputs[0]->data[i];
        }
        grad_info->inputs[0]->backwards();
        grad_info->inputs[1]->backwards();
    }else if (grad_info->operation == "/") {
        for (int i = 0; i < num_elements(); ++i) {
            grad_info->inputs[0]->grad[i] += grad[i] / grad_info->inputs[1]->data[i];
            grad_info->inputs[1]->grad[i] -= grad[i] * data[i] / (grad_info->inputs[1]->data[i] * grad_info->inputs[1]->data[i]);
        }
        grad_info->inputs[0]->backwards();
        grad_info->inputs[1]->backwards();
    }else if (grad_info->operation == "matmul") {
        matmul_der(grad_info->tensor);
        grad_info->inputs[0]->backwards();
        grad_info->inputs[1]->backwards();
    }else{
    std::cout<< grad_info->inputs[0]->check_gradinfo()<<std::endl;
    std::cout<< grad_info->inputs[1]->check_gradinfo()<<std::endl;
    std::cout<<grad_info->operation<<std::endl;}
}

void Tensor::clear_grad() {
    // Set all gradient values to 0.0
    std::fill(grad.begin(), grad.end(), 0.0);

    // If there is no grad_info or inputs, return immediately
    if (!grad_info) {
        return;
    }

    // Recursively clear gradients of the inputs if they exist
    if (grad_info->inputs[0]) {
        grad_info->inputs[0]->clear_grad();
    }
    if (grad_info->inputs[1]) {
        grad_info->inputs[1]->clear_grad();
    }
}


std::vector<float> Tensor::get_grad() const{
    return grad;
}


 void Tensor::set_grad(int x){
    std::fill(grad.begin(), grad.end(), x);
 }

void Tensor::set_grad(std::vector<float> x){
    grad = x;
 }

Tensor Tensor::T() {
    if (shape.size() != 2) {
        std::cout<<' '<<shape.size()<<'\n';
        throw std::invalid_argument("Tensor must be 2D to transpose.");
    }

    Tensor transposed_tensor({shape[1], shape[0]}, 0.0, true); // Initialize with zeros and mark for gradient
    for (int i = 0; i < (shape[0]); ++i) {
        for (int j = 0; j < (shape[1]); ++j) {
            transposed_tensor[{j, i}] = (*this)[{i, j}]; // Transpose data
            //transposed_tensor.grad[get_index({j, i})] = grad[get_index({i, j})]; // Transpose gradient
        }
    }

    return transposed_tensor;
}


 Tensor Tensor::matmul(Tensor& T2, bool t) {
    Tensor T1 = *this;
    if (t) {
        T2 = T2.T(); // Transpose T2 if t is true
    }
    // Check dimensions
    if (T1.shape.size() != 2 || T2.shape.size() != 2 || T1.shape[1] != T2.shape[0]) {
        std::cout<<T1.shape[1]<<' '<<T2.shape[0]<<std::endl;
        throw std::invalid_argument("Invalid dimensions for matrix multiplication.");
    }

    int m = T1.shape[0];
    int n = T1.shape[1];
    int p = T2.shape[1];

    Tensor result({m, p}, 0.0, T1.requires_grad || T2.requires_grad);

    // Perform matrix multiplication
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += T1[{i, k}] * T2[{k, j}];
            }
            result[{i, j}] = sum;
        }
    }
    // Gradient computation for backpropagation
    if (T1.requires_grad || T2.requires_grad) {
        result.grad_info = new GradInfo(&result, "matmul", std::vector<Tensor*>{const_cast<Tensor*>(this), const_cast<Tensor*>(&T2)});
    }

    return result;
}

void Tensor::matmul_der(Tensor* grad_result) {
    if (!requires_grad || !grad_info) {
        return;
    }
    Tensor* T1 = grad_info->inputs[0];
    Tensor* T2 = grad_info->inputs[1];

    // Gradient with respect to T1
    if (T1) {
        Tensor grad_T1 = grad_result->matmul(*T2, true); // Transpose T2 for gradient computation
        T1->set_grad(grad_T1.getData());
    }
    // Gradient with respect to T2
    if (T2) {
        Tensor grad_T2 = T1->T().matmul(*grad_result); // Transpose T1 for gradient computation
         T2->set_grad(grad_T2.getData());
    }
  
}