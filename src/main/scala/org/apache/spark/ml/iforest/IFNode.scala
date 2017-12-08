package org.apache.spark.ml.iforest

sealed abstract class IFNode extends Serializable {
}

/**
  * Data Structure for Isolation Forest Internal Node
  * @param leftChild
  * @param rightChild
  * @param featureIndex
  * @param featureValue
  */
class IFInternalNode (
    val leftChild: IFNode,
    val rightChild: IFNode,
    val featureIndex: Int,
    val featureValue: Double) extends IFNode {
}

class IFLeafNode (
    val numInstance: Long) extends IFNode {
}
