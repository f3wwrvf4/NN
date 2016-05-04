
#include "NN_net.h"
#include <iostream>
#include <fstream>
#include <vector>

namespace NN
{

Network::Network(int* L, int l_num) :
  L_NUM(l_num),
  W_NUM(l_num - 1),
  I_NUM(l_num),
  L_LAST(l_num - 1),
  D_NUM(l_num - 1),
  D_LAST((l_num - 1) - 1)
{

  weights = new Matrix*[W_NUM];
  inputs = new Vector*[I_NUM];
  backs = new Vector*[I_NUM];
  deltas = new Vector*[D_NUM];

  for (int i = 0; i < I_NUM; ++i) {
    int sz;
    if (i == L_LAST) {
      sz = L[i];
    } else {
      sz = L[i] + 1; // weight‚ðÜ‚èž‚Þ
    }
    inputs[i] = new Vector(sz);
    backs[i] = new Vector(sz);
  }

  for (int i = 0; i < W_NUM; ++i) {
    const int szcol = inputs[i]->size();
    const int szrow = inputs[i + 1]->size();

    weights[i] = new Matrix(szrow, szcol);

    float v = 1.0f;
    for (int r = 0; r < szcol; ++r) {
      for (int c = 0; c < szrow; ++c) {
        (*weights[i])(c, r) = (rand() / (float)RAND_MAX) - 0.5f;
      }
    }
  }

  for (int i = 0; i < D_NUM; ++i) {
    deltas[i] = new Vector(inputs[i + 1]->size());
  }
}
Network::~Network()
{
  for (int i = 0; i < I_NUM; ++i) {
    delete inputs[i];
  }
  delete[] inputs;

  for (int i = 0; i < W_NUM; ++i) {
    delete weights[i];
  }
  delete[] weights;

  for (int i = 0; i < D_NUM; ++i) {
    delete deltas[i];
  }
  delete[] deltas;
}

void Network::train(const TrainData* td_tbl, int num)
{
  const int DATA_COUNT = num;
  const int TRAIN_COUNT = 10000000;

  for (int iTrainCount = 0; iTrainCount < TRAIN_COUNT; ++iTrainCount) {
    float error = 0;
    for (int iDataCount = 0; iDataCount < DATA_COUNT; ++iDataCount) {
      const TrainData& data = td_tbl[iDataCount];
      {
        Vector& y = *inputs[0];
        for (int i = 0; i < y.size() - 1; ++i) {
          y(i) = data.input[i];
        }
      }

      // forwad
      feedForward();

      // Œë·‚Ì•]‰¿
      {
        const Vector& out = *inputs[L_LAST];
        for (int i = 0; i < out.size(); ++i) {
          error += ((data.output[i] - out(i))*(data.output[i] - out(i)));
        }
      }

      // back
      {
        backPropagate(data);
      }
    }

    if ((iTrainCount % 100) == 0) {
      std::cout << "Train Count:" << iTrainCount
        << "  error:" << error << std::endl;
    }

    if (error < 0.01f) {
      std::cout << "err:" << error << std::endl;
      break;
    }
  }
}

void Network::feedForward()
{
  // ‡“`”d
  for (int iL = 0; iL < L_NUM - 1; ++iL) {
    Vector& y = (*inputs[iL]);
    Matrix& w = (*weights[iL]);
    Vector& o = (*inputs[iL + 1]);
    if (iL < L_LAST) {
      const int last = y.size() - 1;
      y(last) = 1.0f;  // weight‚ð“ü—Í‚Æ‚µ‚Ä•]‰¿‚·‚é
    }
    NN::Mul(y, w, o);
    for (int i = 0; i < o.size(); ++i) {
      o(i) = sigmoid(o(i));
    }
  }
}

void Network::backPropagate(const TrainData& data)
{
  {
    // o—Í‘w
    {
      const Vector& out = *inputs[L_LAST];
      Vector& delta = *deltas[D_LAST];
      for (int i = 0; i < out.size(); ++i) {
        delta(i) = (out(i) - data.output[i]) * (1.0f - out(i)) * out(i);
      }
    }

    // ‰B‚ê‘w
    for (int i = W_NUM - 1; 0 < i; --i) {
      const int layer = i;
      const Vector& hidden = *inputs[layer];
      const Matrix& weight = *weights[layer];
      const Vector& prev_delta = *deltas[layer];
      Vector& hid_back = *deltas[layer - 1];
      Mul(weight, prev_delta.t(), hid_back);
      for (int i = 0; i < hid_back.size(); ++i) {
        hid_back(i) = hid_back(i) * (1.0f - hidden(i)) * hidden(i);
      }
    }

    // d‚Ý‚ðC³
    const float eps = 0.1f;
    for (int i = 0; i < W_NUM; ++i) {
      Matrix& weigh = *weights[i];
      const Vector& back = *deltas[i];
      const Vector& inp = *inputs[i];

      for (int iRow = 0; iRow < weigh.row(); ++iRow) {
        for (int iCol = 0; iCol < weigh.col(); ++iCol) {
          float b = back(iRow);
          float in = inp(iCol);
          weigh(iRow, iCol) -= eps*in*b;
        }
      }
    }
  }
}

std::vector<float> Network::eval(std::vector<float> input)
{
  Vector& y = *inputs[0];
  for (int i = 0; i < y.size() - 1; ++i) {
    y(i) = input[i];
  }
  feedForward();
  return (*inputs[L_LAST]).vec();
}

void Network::save(const char* fpath)
{
  std::ofstream ofs(fpath);

  // input
  ofs << I_NUM << std::endl;
  for (int i = 0; i < I_NUM; ++i) {
    int sz = inputs[i]->size();
    if (i != (I_NUM - 1)) {
      sz -= 1;
    }
    ofs << sz << " ";
  }
  ofs << std::endl;

  // weight
  ofs << W_NUM << std::endl;
  for (int i = 0; i < W_NUM; ++i) {
    ofs << (*weights[i]);
  }
  ofs << std::endl;
}

Network* Network::CreateFromFile(const char* fpath)
{
  std::ifstream ifs(fpath);
  int i_num;
  ifs >> i_num;
  int* ary = new int[i_num];
  for (int i = 0; i < i_num; ++i) {
    ifs >> ary[i];
  }
  Network* net = new Network(ary, i_num);
  delete[] ary;

  int w_num;
  ifs >> w_num;
  for (int i = 0; i < w_num; ++i) {
    ifs >> *net->weights[i];
  }
  return net;
}

}
