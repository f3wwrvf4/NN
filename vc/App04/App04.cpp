// App01.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include <windows.h>


#include "NN_net.h"
#include "NN_mnist.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <new>
#include <string>
#include <list>

namespace NN
{

struct Layer
{
  const Matrix& forward(const Matrix& input)
  {
    if (next)
      NN::Weight::calcLogisticMid(weight, input, out, differ);
    else
      NN::Weight::calcLogisticOut(weight, input, out);

    input.t(input_trn);

    return out;
  }
  const Matrix& back(const Matrix& out_delta)
  {
    NN::Mul(out_delta, input_trn, rdw);

    if (prev) {
      (weight).t(weight_trn);
      NN::Mul(weight_trn, out_delta, delta);
      NN::Hadamard(prev->differ, delta, delta);
    }

    const float eps = 0.1f;
    NN::Mul(-eps, rdw, rdw);
    NN::Add(weight, rdw, weight);

    return delta;
  }

  Layer(int _batch_num, int _node1, int _node2):
    input_trn(_node1, _batch_num),
    out(_batch_num, _node2),
    out_trn(_node2, _batch_num),
    weight(_node1, _node2),
    weight_trn(_node2, _node1),
    delta(_batch_num, _node1),
    differ(_batch_num, _node2),
    rdw(_node1, _node2)
  {
    batch_num = _batch_num;
    node1 = _node1;
    node2 = _node2;
    weight.random();
    prev = next = 0;
  }
  ~Layer()
  {
  }

  int batch_num;
  int node1, node2;

  NN::Matrix input_trn;

  NN::Matrix out;
  NN::Matrix out_trn;

  NN::Matrix weight;
  NN::Matrix weight_trn;

  NN::Matrix delta;
  NN::Matrix differ;
  NN::Matrix rdw;

  Layer* prev;
  Layer* next;
};


} // namespace NN


int main()
{
  struct
  {
    float ti[3];
    float to[2];
  }td[] = {
    {{0, 0, 0}, {1, 1}},
    {{0, 0, 1}, {1, 0}},
    {{0, 1, 0}, {1, 1}},
    {{0, 1, 1}, {1, 0}},
    {{1, 0, 0}, {1, 1}},
    {{1, 0, 1}, {1, 0}},
    {{1, 1, 0}, {0, 1}},
    {{1, 1, 1}, {0, 1}},
  };

  const int InputNum = 3;
  const int MidNum = 5;
  const int OutNum = 2;

  const int Node1 = InputNum + 1;
  const int Node2 = MidNum + 1;
  const int Node3 = OutNum;

  const int bn = 4;

  NN::Layer* layer_1 = new NN::Layer(bn, Node1, Node2);
  NN::Layer* layer_2 = new NN::Layer(bn, Node2, Node3);

  layer_1->next = layer_2;
  layer_2->prev = layer_1;

  DWORD msec1 = GetTickCount();

  NN::Matrix input(bn, Node1);
  NN::Matrix delta_o(bn, Node3);

  float error=0;
  int iTrain = 0;
  while (true) {
    ++iTrain;
    if ((iTrain & 1) == 1)
      error = 0;

    for (int i = 0; i < bn; ++i) {
      int tid = i + (iTrain & 1 ? 0 : bn);
      for (int j = 0; j < InputNum; ++j) {
        input(i, j) = td[tid].ti[j];
      }
      input(i, InputNum) = 1.0f;
    }

    const NN::Matrix& hidden = layer_1->forward(input);
    const NN::Matrix& out = layer_2->forward(hidden);

    // error
    for (int i = 0; i < bn; ++i) {
      for (int j = 0; j < OutNum; ++j) {
        int tid = i + (iTrain & 1 ? 0 : bn);
        const float err = NN::Square(td[tid].to[j] - out(i, j));
        error += err;
        delta_o(i, j) = out(i, j) - td[tid].to[j];
      }
    }
    if (error < 0.01f) {
      for (int i = 0; i < bn; ++i) {
        for (int j = 0; j < OutNum; ++j) {
          int tid = i + (iTrain & 1 ? 0 : bn);
          float t = td[tid].to[j];
          float o = out(i, j);

          std::cout << t << " - " << o << std::endl;
        }
      }
      std::cout << iTrain << ":" << error << std::endl;
      break;
    }

    // back
    const NN::Matrix& delta_h = layer_2->back(delta_o);
    const NN::Matrix& delta_i = layer_1->back(delta_h);
  }

  DWORD msec2 = GetTickCount();
  std::cout << "sec: " << msec2 - msec1;

  getchar();
  return 0;
}
