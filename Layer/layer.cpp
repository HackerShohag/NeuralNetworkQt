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

#include <QtMath>
#include <QDebug>
#include <QRandomGenerator>

#include "Layer"

Layer::Layer()
{
//    qDebug() << Q_FUNC_INFO;
    this->numNodesIn = 0;
    this->numNodesOut = 0;
}

Layer::Layer(int numNodesIn, int numNodesOut)
{
    qDebug() << Q_FUNC_INFO;
    this->numNodesIn = numNodesIn;
    this->numNodesOut = numNodesOut;

    weightedInputs = new QVector<double>();
    activations = new QVector<double>();

    InitializeRandomWeights();
}

Layer::~Layer()
{
//    qDebug() << Q_FUNC_INFO;
}

QVector<double> *Layer::CalculateOutputs(QVector<double> *inputs)
{
    qDebug() << Q_FUNC_INFO;
    QVector<double> *activationValues = new QVector<double>();
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
        double weightedInput = biases->at(nodeOut);
        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
            weightedInput += inputs->at(nodeIn) * weights->at(nodeIn).at(nodeOut);
            weightedInputs->append(weightedInput);
        }
        double activationValue = ActivationFunction(weightedInput);
        activations->append(activationValue);
        activationValues->insert(nodeOut, activationValue);
    }
    return activationValues;
}

void Layer::ApplyGradients(double learnRate)
{
    qDebug() << Q_FUNC_INFO;
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
    {
        (*biases)[nodeOut] -= costGradientBiases->at(nodeOut) * learnRate;
        //biases->insert(nodeOut, biases->at(nodeOut) - costGradientBiases->at(nodeOut) * learnRate);
        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
            (*weights)[nodeIn][nodeOut] = costGradientWeights->at(nodeIn).at(nodeOut) * learnRate;
        }
    }
}

void Layer::InitializeRandomWeights()
{
    qDebug() << Q_FUNC_INFO;
    weights = new QVector<QVector<double>>(numNodesIn);
    for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
    {
        QVector<double> randomWights(numNodesOut);
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double randomValue = (QRandomGenerator::global()->generateDouble() * 2 - 1) / qSqrt(numNodesIn);
            randomWights[nodeOut] = randomValue;
        }
        (*weights)[nodeIn] = randomWights;
    }
}

double Layer::ActivationFunction(double weightedInput)
{
    qDebug() << Q_FUNC_INFO;
    return 1 / (1 + qExp(-weightedInput));
}

double Layer::ActivationFunctionDerivative(double weightedInput)
{
    qDebug() << Q_FUNC_INFO;
    double activation = ActivationFunction(weightedInput);
    return activation * (1 - activation);
}

double Layer::NodeCost(double outputActivation, double expectedOutput)
{
    qDebug() << Q_FUNC_INFO;
    double error = outputActivation - expectedOutput;
    return error * error;
}

QVector<double> *Layer::CalculateOutputLayerNodeValues(QVector<double> *expectedOutputs)
{
    nodeValues =  new QVector<double>(expectedOutputs->length());
    for (int i = 0; i < nodeValues->length(); i++){
        // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
        double costDerivative = NodeCostDerivative(activations->at(i), expectedOutputs->at(i));
        double activationDerivative = ActivationFunctionDerivative(weightedInputs->at(i));
        (*nodeValues)[i]  = activationDerivative * costDerivative;
    }
    return nodeValues;
}

void Layer::UpdateGradients(QVector<double> *nodeValues)
{
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
    {
        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
        {
            // Evaluate the partial derivative: cost / weight of current connection
            double derivativeCostWrtweight = weightedInputs->at(nodeIn) * nodeValues->at(nodeOut);
            // The costGradientW array stores these partial derivatives for each weight.
            //Note: the derivative is being added to the array here because ultimately we want
            //to calculate the average gradient across all the data in the training batch
            (*costGradientWeights)[nodeIn][nodeOut ] += derivativeCostWrtweight;
        }
        // Evaluate the partial derivative: cost / bias of the current node
         double derivativeCostWrtBias = 1 * nodeValues->at(nodeOut);
        costGradientBiases[nodeOut] += derivativeCostWrtBias;
    }
}

QVector<double> *Layer::CalculateHiddenLayerNodeValues(Layer *oldLayer, QVector<double> *oldNodeValues)
{
    QVector<double> *newNodeValues = new QVector<double>(numNodesOut);
    QVector<QVector<double>> *oldLayerWeights = oldLayer->getWeights();
    for (int newNodeIndex = 0; newNodeIndex < newNodeValues->length(); newNodeIndex++)
    {
        double newNodeValue = 0;
        for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues->length(); oldNodeIndex++)
        {
            // Partial derivative of the weighted input with respect to the input
            double weightedInputDerivative = (*oldLayerWeights)[newNodeIndex][oldNodeIndex];
            newNodeValue += weightedInputDerivative * oldNodeValues->at(oldNodeIndex);
        }
        newNodeValue *= ActivationFunctionDerivative(weightedInputs->at(newNodeIndex));
        (*newNodeValues)[newNodeIndex] = newNodeValue;
    }
    return newNodeValues;
}

double Layer::NodeCostDerivative(double outputActivation, double expectedOutput)
{
    qDebug() << Q_FUNC_INFO;
    return 2 * (outputActivation - expectedOutput);
}

int Layer::getNumNodesIn() const
{
    return numNodesIn;
}

int Layer::getNumNodesOut() const
{
    return numNodesOut;
}

QVector<QVector<double> > *Layer::getCostGradientWeights() const
{
    return costGradientWeights;
}

QVector<double> *Layer::getCostGradientBiases() const
{
    return costGradientBiases;
}

QVector<QVector<double> > *Layer::getWeights() const
{
    return weights;
}

QVector<double> *Layer::getBiases() const
{
    return biases;
}
