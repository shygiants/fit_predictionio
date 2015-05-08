package kr.ac.korea.ee.fit

import org.scalatest.FlatSpec
import org.scalatest.Matchers

class PreparatorTest
  extends FlatSpec with EngineTestSparkContext with Matchers {

  val preparator = new Preparator()
  val users = Map(
    "u0" -> User(),
    "u1" -> User()
  )

  val items = Map(
    "i0" -> Item(categories = Some(List("c0", "c1"))),
    "i1" -> Item(categories = None)
  )

  val rate = Seq(
    RateEvent("u0", "i0", 3.5, 1000010),
    RateEvent("u0", "i1", 0.5, 1000020),
    RateEvent("u1", "i1", 5.0, 1000030)
  )

  // simple test for demonstration purpose
  "Preparator" should "prepare PreparedData" in {

    val trainingData = new TrainingData(
      users = sc.parallelize(users.toSeq),
      items = sc.parallelize(items.toSeq),
      rateEvents = sc.parallelize(rate.toSeq)
    )

    val preparedData = preparator.prepare(sc, trainingData)

    preparedData.users.collect should contain theSameElementsAs users
    preparedData.items.collect should contain theSameElementsAs items
    preparedData.rateEvents.collect should contain theSameElementsAs rate
  }
}
