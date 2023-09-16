/*
 * Copyright (C) 2023  Abdullah AL Shohag
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; version 3.
 *
 * raven.downloader is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <Layer>

class NeuralNetwork
{
    QVector<Layer> *layers;

public:
    struct DataPoint{
        template<typename T> QVector<T> setInputs() {

        }
//        *inputs;
        QVector<double> *expectedOutputs;
    };

    NeuralNetwork(QVector<qint16> layerSizes);
    ~NeuralNetwork();
    QVector<double> *CalculateOutputs(QVector<double> *inputs);
    int Classify(QVector<double> *inputs);
    double Cost(DataPoint *dataPoint);
    double Cost(QVector<DataPoint> *data);
    void Learn(QVector<DataPoint> *trainingData, double learnRate);
    void ApplyAllGradients(double learnRate);
    void UpdateAllGradients(DataPoint dataPoint);
};

#endif // NEURALNETWORK_H
