#include <iostream>
#include <string>
#include <vector>
#include "Tensor.cpp"

int main() {
    std::vector<std::vector<std::vector<float>>> vec3d = {
        { {1.0, 2.0}, {3.0, 4.0} },
        { {5.0, 6.0}, {7.0, 8.0} },
        { {9.0, 10.0}, {11.0, 12.0} },
        { {13.0, 14.0}, {15.0, 16.0} }
    };

    std::vector<std::vector<std::vector<std::vector<float>>>> vec4d = {
        {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}},
        {{{9.0, 10.0}, {11.0, 12.0}}, {{13.0, 14.0}, {15.0, 16.0}}}
    };

    std::vector<float> vec1d = {1.0f,2.0f,3.0f,4.0f,5.0f};

    std::vector<std::vector<float>> vec1 ={{1,2},{3,4},{5,6}}; //3x2 matrix
    std::vector<std::vector<float>> vec2 ={{1,2,3},{4,5,6}}; //2x3 matrix

    Tensor T1(vec1);
    Tensor T2(vec2);
  
    Tensor T4(vec3d);
    Tensor T5(vec3d);

    Tensor T7(vec4d);
    Tensor T8(vec4d);

    Tensor T3 = T1.matmul(T2); // should return a 3x3 matrix with values {{9 12 15} {19 26 33} {29 40 51}}
    Tensor T6 = T4 + T5; // should return a 3d matrix with values {{{2 4} {6 8}} {{10 12} {14 16}} {{18 20} {22 24}} {{26 28} {30 32}} }
    Tensor T9 = T7 * T8; //should return 4d matrix with values 1 4 9 16 25 36 49 64 81 100 121 144 169 196 225 256 

 
    std::cout<<"T3:"<<'\n';
    for(float i:T3.getData()){std::cout<<i<<' ';}
    std::cout<<'\n';
    std::cout<<"T4:"<<'\n';
    for(float i:T6.getData()){std::cout<<i<<' ';}
    std::cout<<'\n';
    std::cout<<"T9:"<<'\n';
    for(float i:T9.getData()){std::cout<<i<<' ';}
    std::cout<<'\n';
    std::cout<<'\n';


    T3.set_grad();
    T6.set_grad();
    T9.set_grad();
    
    T6.backwards();
    T3.backwards();
    T9.backwards();

    std::cout<<"T1: ";
    for(float i:T1.get_grad()){std::cout<<i<<' ';}
    std::cout<<'\n';
    std::cout<<"T2: ";
    for(float i:T2.get_grad()){std::cout<<i<<' ';}
    std::cout<<'\n'; 
    std::cout<<"T4: ";
    for(float i:T4.get_grad()){std::cout<<i<<' ';}
    std::cout<<'\n';
    std::cout<<"T5: ";
    for(float i:T5.get_grad()){std::cout<<i<<' ';}
    std::cout<<'\n'; 
    std::cout<<"T7: ";
    for(float i:T7.get_grad()){std::cout<<i<<' ';}
    std::cout<<'\n';
    std::cout<<"T8: ";
    for(float i:T8.get_grad()){std::cout<<i<<' ';}  

    return 0;
}