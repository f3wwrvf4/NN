#pragma once

#include "NN_math.h"
#include <vector>

namespace NN
{

struct TrainData
{
  TrainData(std::vector<float> inp, std::vector<float> out)
  {
    input = inp;
    output = out;
  }
  std::vector<float> input;
  std::vector<float> output;
};

class Network
{
  const int W_NUM;
  const int I_NUM;
  const int L_LAST;
  const int L_NUM;
  const int D_NUM;
  const int D_LAST;

  Matrix** weights;
  Vector** inputs;
  Vector** backs;
  Vector** deltas;

public:
  Network(int*L, int l_num);
  ~Network();

  void train(TrainData* td_tbl, int num);
  std::vector<float> eval(std::vector<float> input);

  void save(const char* fpath);
  static Network* CreateFromFile(const char* fpath);

  float sigmoid(float v)
  {
    return 1.0f / (1.0f + exp(-v));
  }

protected:
  void feedForward();
  void backPropagate(const TrainData& data);

}; // Network


}