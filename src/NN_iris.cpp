#include "NN_iris.h"

#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

namespace NN
{
Iris::Iris()
{
}

Iris::~Iris()
{
}

void Iris::LoadData()
{
  // image
  {
    ifstream ifs("../data/iris.txt", ios::in);
    assert(!ifs.fail());

    char c;
    char str[32];

    int line = 0;
    int count[2][3] = {};
    while (!ifs.eof()) {
      float f[DataSize] = {};
      float o[LabelSize] = {};
      ifs >> f[0] >> c >> f[1] >> c >> f[2] >> c >> f[3] >> c >> str;

      int id = line & 1;
      bool fail = false;
      if (strcmp(str, "setosa") == 0) {
        o[0] = 1.0f;
        ++count[id][0];
      } else if (strcmp(str, "versicolor") == 0) {
        o[1] = 1.0f;
        ++count[id][1];
      } else if (strcmp(str, "virginica") == 0) {
        o[2] = 1.0f;
        ++count[id][2];
      } else {
        fail = true;
      }
      if (!fail) {
        if (line & 1) {
          trainData.push_back(Data(f, o));
        } else {
          testData.push_back(Data(f, o));
        }
      }
      ++line;
    }
  }
}

void Iris::shuffle()
{
  random_shuffle(trainData.begin(), trainData.end());
}

int Iris::GetTrainDataCount() const
{
  return (int)trainData.size();
}

const Matrix& Iris::GetTrainInputData(int idx, int count, NN::Matrix* mat) const
{
  _ASSERT(count == mat->row());
  _ASSERT((DataSize + 1) == mat->col());
  _ASSERT((idx + count) <= trainData.size());

  for (int i = idx; i < idx + count; ++i) {
    int x = i%count;
    for (int j = 0; j < DataSize; ++j) {
      (*mat)(x, j) = trainData[i].input[j];
    }
    (*mat)(x, DataSize) = 1.0f;
  }
  return *mat;
}
const Matrix& Iris::GetTrainOutputData(int idx, int count, NN::Matrix* mat) const
{
  _ASSERT(count == mat->row());
  _ASSERT(LabelSize == mat->col());
  _ASSERT((idx + count) <= trainData.size());

  for (int i = idx; i < idx + count; ++i) {
    int x = i%count;
    for (int j = 0; j < LabelSize; ++j) {
      (*mat)(x, j) = trainData[i].out[j];
    }
  }
  return *mat;
}

int Iris::GetTestDataCount() const
{
  return (int)testData.size();
}

const Matrix& Iris::GetTestInputData(int idx, int count, NN::Matrix* mat) const
{
  _ASSERT(count == mat->row());
  _ASSERT((DataSize + 1) == mat->col());
  _ASSERT((idx + count) <= testData.size());

  for (int i = idx; i < idx + count; ++i) {
    int x = i%count;
    for (int j = 0; j < DataSize; ++j) {
      (*mat)(x, j) = testData[i].input[j];
    }
    (*mat)(x, DataSize) = 1.0f;
  }
  return *mat;
}
const Matrix& Iris::GetTestOutputData(int idx, int count, NN::Matrix* mat) const
{
  _ASSERT(count == mat->row());
  _ASSERT(LabelSize == mat->col());
  _ASSERT((idx + count) <= testData.size());

  for (int i = idx; i < idx + count; ++i) {
    int x = i%count;
    for (int j = 0; j < LabelSize; ++j) {
      (*mat)(x, j) = testData[i].out[j];
    }
  }
  return *mat;
}

}