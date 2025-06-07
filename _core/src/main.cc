// #include "../include/tensor.hpp"
#include "../include/tensor.hpp"

void test() {
  Tensor<double> t1 = Tensor<double>::rand({10, 10});
  Tensor<double> t2 = Tensor<double>::rand({10, 10});
  Tensor<double> t3 = t1.add(t2);
  t1.print(true);
  t2.print(true);
  t3.print();
}

void init() {
  auto cek_valid = [](Tensor<int> &x) {
    std::cout << (x.is_valid() ? "Yes Valid" : "Invalid") << std::endl;
  };
  // Owned Buffer
  Tensor<int> t1({2, 2}, {1, 2, 3, 4});
  t1.info();
  cek_valid(t1);
  t1.print(1);

  // External Buffer
  int external[4] = {10, 20, 30, 40};
  Tensor<int> t2({2, 2}, external, 4);
  t2.info();
  cek_valid(t2);
  t2.print(1);

  // Copy Constructor
  Tensor<int> t3 = t2;
  t3.info();
  cek_valid(t3);
  t3.print(1);

  Tensor<int> t4 = std::move(t1);
  t4.info();
  cek_valid(t4);
  t4.print(1);

  cek_valid(t1);
  t1.print();
}

void vector() {
  // Tensor<int> t1 = Tensor<int>::randint({2, 0}, 0, 10);
  Tensor<int> t1({2}, {1, 2});
  t1.info();
  t1.print();
  Tensor<int> t2 = t1.reshape({2, 1});

  t2.info();
  t2.print();

  Tensor<int> t3 = t2.transpose();
  t3.info();
  t3.print();
}

void transpose() {
  Tensor<int> t1 = Tensor<int>::randint({10, 2}, 0, 10);
  t1.print();
  Tensor<int> t2 = t1.transpose();
  t2.print();
}

void ops() {
  Tensor<int> t1 = Tensor<int>::randint({10, 10}, 0, 10);
  t1.flatten();
  t1.info();
}

int main() {
  ops();
  return 0;
}
