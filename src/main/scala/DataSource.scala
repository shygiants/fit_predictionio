package kr.ac.korea.ee.fit

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
import io.prediction.data.storage.Event
import io.prediction.data.store.PEventStore

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger

case class DataSourceParams(appName: String) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {

    // create a RDD of (entityID, User)
    val usersRDD: RDD[(String, User)] = PEventStore.aggregateProperties(
      appName = dsp.appName,
      entityType = "user"
    )(sc).map { case (entityId, properties) =>
      val user = try {
        User()
      } catch {
        case e: Exception => {
          logger.error(s"Failed to get properties ${properties} of" +
            s" user ${entityId}. Exception: ${e}.")
          throw e
        }
      }
      (entityId, user)
    }.cache()

    // create a RDD of (entityID, Item)
    val itemsRDD: RDD[(String, Item)] = PEventStore.aggregateProperties(
      appName = dsp.appName,
      entityType = "item"
    )(sc).map { case (entityId, properties) =>
      val item = try {
        // Assume categories is optional property of item.
        Item(categories = properties.getOpt[List[String]]("categories"))
      } catch {
        case e: Exception => {
          logger.error(s"Failed to get properties ${properties} of" +
            s" item ${entityId}. Exception: ${e}.")
          throw e
        }
      }
      (entityId, item)
    }.cache()

    val eventsRDD: RDD[Event] = PEventStore.find(
      appName = dsp.appName,
      entityType = Some("user"),
      eventNames = Some(List("rate")),
      // targetEntityType is optional field of an event.
      targetEntityType = Some(Some("item")))(sc)
      .cache()

    // get all "user" "rate" "item" events
    val rateEventsRDD: RDD[RateEvent] = eventsRDD
      .filter { event => event.event == "rate" }
      .map { event =>
        try {
          RateEvent(
            user = event.entityId,
            item = event.targetEntityId.get,
            rating = event.properties.get[Double]("rating"),
            t = event.eventTime.getMillis
          )
        } catch {
          case e: Exception =>
            logger.error(s"Cannot convert ${event} to RateEvent." +
              s" Exception: ${e}.")
            throw e
        }
      }
      
    new TrainingData(
      users = usersRDD,
      items = itemsRDD,
      rateEvents = rateEventsRDD
    )
  }
}

case class User()

case class Item(categories: Option[List[String]])

case class RateEvent(user: String, item: String, rating: Double, t: Long)

class TrainingData(
  val users: RDD[(String, User)],
  val items: RDD[(String, Item)],
  val rateEvents: RDD[RateEvent]
) extends Serializable {
  override def toString = {
    s"users: [${users.count()} (${users.take(2).toList}...)]" +
    s"items: [${items.count()} (${items.take(2).toList}...)]" +
    s"rateEvents: [${rateEvents.count()}] (${rateEvents.take(2).toList}...)"
  }
}
