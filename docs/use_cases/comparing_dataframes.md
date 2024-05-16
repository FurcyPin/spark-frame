## What is data-diff ?

DataFrame (or table) comparison is the most important feature that any SQL-based OLAP engine
should have. I personally use it all the time whenever I work on a data pipeline, and
I think it is so useful and powerful that it should be a built-in feature of any
data pipeline development tool, just like `git diff` is the most important feature when
you use git.

![type:video-autoplay](/spark-frame/images/carburant_diff.mp4)
<figcaption>Here is an example of data diff</figcaption>

## What does it do ?

Simple: it compares two SQL tables (or DataFrames), and gives you a detailed summary
of what changed between the two, letting you perform in-depth analysis when required. 


## Why is it useful ?

During the past few decades, code diff, alongside automated testing, has become the cornerstone tool 
used for implementing coding best-practices, like versioning and code reviews. 
No sane developer would ever consider using versioning and code reviews if code-diff wasn't possible.


![](/spark-frame/images/git_diff.png)
<figcaption>Here is an example of code diff, everyone should be quite familiar with them already.</figcaption>



When manipulating complex data pipelines, in particular with code written in SQL or DataFrames, it quickly
becomes extremely challenging to anticipate and make sure that a change made to the code will not have any
unforeseen side effects.


## How to get started ?

Here is a minimal example of how you can use it : 


First, create a PySpark job with spark-frame and data-diff-viewer as dependencies 
  (check this project's README.md to know which versions of data-diff-viewer are compatible with spark-frame) 

Then run a PySpark job like this one:
``` py title="data_diff.py" linenums="1" hl_lines="11-14"
from pyspark.sql import SparkSession
from spark_frame.data_diff import compare_dataframes

# Instantiate the SparkSession the way you like
spark = SparkSession.builder.appName("data-diff").getOrCreate()

# Read or compute the two DataFrames you want to compare together
df1 = spark.read.parquet("...")
df2 = spark.table("...")

# Compare the two DataFrames
diff_result = compare_dataframes(df1, df2, join_cols=[...])
# Export the diff result as HTML
diff_result.export_to_html(output_file_path="diff_report.html")
```
 
And that's it! After the job has run, you should get an HTML report at the location specified with `output_file_path`.

The only parameter to be wary of are:

### join_cols
`join_cols` indicates the list of column names that will be used to join the two DataFrame together for the 
comparison. This set of columns should follow an unicity constraint in both DataFrames to prevent a 
combinatorial explosion.

!!! success "Features"

    - If `join_cols` is not set, the algorithm will try to infer one column automatically, but it can only detect single 
      columns, if the DataFrames require multiple columns to be joined together, the automatic detection will not work.
    - The algorithm comes with a safety mechanism to avoid performing joins that would lead to a combinatorial explosion
      when the join_cols are incorrectly chosen.

### output_file_path
The path where the HTML report should be written.
    
!!! success "Features"
    This method uses Spark's FileSystem API to write the report.
    This means that `output_file_path` behaves the same way as the path argument in `df.write.save(path)`:

    - It can be a fully qualified URI pointing to a location on a remote filesystem
      (e.g. "hdfs://...", "s3://...", etc.), provided that Spark is configured to access it
    - If a relative path with no scheme is specified (e.g. `output_file_path="diff_report.html"`), it will
      write on Spark's default's output location. For example:
        - when running locally, it will be the process current working directory.
        - when running on Hadoop, it will be the user's home directory on HDFS.
        - when running on the cloud (EMR, Dataproc, Azure Synapse, Databricks), it should write on the default remote
          storage linked to the cluster.

**Methods used in this example**

??? abstract "spark_frame.data_diff.compare_dataframes"
    ::: spark_frame.data_diff.compare_dataframes
        options:
            show_root_heading: false
            show_root_toc_entry: false

??? abstract "spark_frame.data_diff.DiffResult.export_to_html"
    ::: spark_frame.data_diff.DiffResult.export_to_html
        options:
            show_root_heading: false
            show_root_toc_entry: false

## How to read it ?

Once the report is generated, you can simply open it with any browser and 
you will see a web page with three sections:

- A title headline

- A schema diff that should look like this:

![](/spark-frame/images/carburant_diff_schema.png)

- And a data-diff that should look like this:

![](/spark-frame/images/carburant_diff_data.png)

The data diff reads like this:

- There is one line per column in the DataFrame. 
- If a column does not exist in the first DataFrame, it will be prefixed with a `+` and the bar will be 
  <span style="color:purple">purple</span>.
- If a column does not exist in the second DataFrame, it will be prefixed with a `-` and the bar will be 
  <span style="color:blue">blue</span>.
- Otherwise, the bar chart represent the percentage of rows that are identical 
  (in <span style="color:green">green</span>) _vs._ the percentage that differ (in <span style="color:red">red</span>).

You can interact with the data diff by mousing over and clicking
on the bar charts:

- If you mouse over a bar chart, a detailed percentage and count of each type of value will appear:
![type:video](/spark-frame/images/data_diff_mousover.mp4)

- If you click on a bar chart, a detail of the most frequent changes and non-changes will appear :
![type:video](/spark-frame/images/data_diff_click.mp4)

- You can then click on the details to display a full example of row comparison where this change/non-change happens:
![type:video](/spark-frame/images/details_click.mp4)
- Clicking on the line that is already selected will make the full example disappear

- On top of the data diff, there are also two buttons: "Hide columns with no change" and "Expand all details", which
  are useful when you need to focus on reviewing all the changes in a data-diff:
![type:video](/spark-frame/images/data_diff_buttons.mp4)

## Examples

### Simple examples

Some simple examples are available in the reference of the method 
[spark_frame.data_diff.compare_dataframes][spark_frame.data_diff.compare_dataframes].

### French gas price

We made a notebook using a real complex use-case using Open Data.
You can open it directly in Google Colab using this link: <a href="https://colab.research.google.com/github/FurcyPin/french-gas-price/blob/main/notebooks/Data_diff_spark-french_gas_prices.ipynb" target="_blank">
![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)
</a>

Or if you don't want to use Google Colab, you can open it 
[directly in github](https://github.com/FurcyPin/french-gas-price/blob/main/notebooks/Data_diff_spark-french_gas_prices.ipynb).

If you're in a hurry, here are two examples of diffs generated with this notebook:

- [Diff obtained in the middle of the cleaning process](/spark-frame/diff_reports/carburants_diff_3.html)
- [Diff obtained at the end, before analysis of the remaining differences](/spark-frame/diff_reports/carburants_diff_7.html)


## What are some common use cases ?

From the simplest to the most complex, data-diff is super useful in many cases :

### Refactoring a single SQL query 

Refactoring a single SQL query is something I do very often when I get my hand on legacy code.
I do it quite often in order to :

- Improve the readability of the code.
- Get to know the code better (_"why is it done like this and not like that ?"_).
- Improving the performances.

Usually, the results of the refactoring process will be a new query which is much cleaner, more efficient, but
produces _exactly the same result_.

That's when data-diff becomes extremely handy, as it can make sure the results are 100% identical.

_Funny story:_ more than once, after refactoring a SQL query, I noticed small differences in the results thanks to data-diff. 
After further inspection I sometimes realized that my refactoring had introduced a new bug, but other times 
I realized that my refactoring actually fixed a bug that was previously existing and went unnoticed until then. :grin:

But of course, what we said also applies to data transformations written with DataFrames, and also to whole pipelines
chaining multiple transformations or queries.

### Refactoring a whole data pipeline

Refactoring a data pipeline is often one of the most daunting and scary tasks that an Analytics Engineer has to carry.
But there are many good reasons to do it, such as:
 
- To reduce code duplication (when you realise that the exact same CTE appears in 3 or 4 different queries) 
- To improve maintainability, readability, stability, documentation, testing: if a query contains 20 CTEs,
  it's probably a good idea to split it in smaller parts, thus making the intermediary results easier to document, 
  test and inspect.
- For many other cases listed below in dedicated items

Here again, when we do so we want to make sure the results are exactly the same as they were before.

### Fixing a bug in a query

Whenever I work on a fix, I use data-diff to make sure that the changes I wrote have the exact effect I was planning to
have on the query **and nothing more**. I also make sure that this change does not impact the tables downstream.

??? Example
    Here is a very simple example of a fix that could go wrong: you notice one of your type has a column `country` 
    that contains the following value counts:

    | direction | count   |
    |-----------|---------|
    | FRANCE    | 124 590 |
    | france    | 4 129   |
    | germany   | 209 643 |
    | GERMANY   | 1 2345  |
    
    !!! note 
        As this is a toy example, we won't go far into the details of why such things happened in the first place. 
        Let's imagine you work for a French company that bought a German company, and you are merging the two information 
        systems, and that at some point the prod platform was updated to make sure only UPPERCASE was used.
        Needless to stress the importance of making sure that your input data is of the highest possible quality 
        using data contracts...
    
    You decide to apply a fix by passing everything to uppercase, adding a nice `"SELECT UPPER(direction) as direction"`
    somewhere in the code, ideally during the cleaning or bronze stage of your data pipeline. (_Needless to say, it would
    be even better if the golden source for your data were fixed and backfilled, or even better if that inconsistency 
    never happened in the first place, but that kind of perfect solution requires a strong commitment at every level of your 
    organisation, and the metrics in your CEO's dashboard needs to be fixed today, not in a few months..._) 
    
    So you implement that fix, and run data-diff to make sure the results are good. You get something like this:
    
        136935 (39.0%) rows are identical
        213772 (61.0%) rows have changed
        0 (0.0%) rows are only in 'left'
        0 (0.0%) rows are only in 'right
        
        Found the following changes:
        +-----------+-------------+---------------------+---------------------------+--------------+
        |column_name|total_nb_diff|left_value           |right_value                |nb_differences|
        +-----------+-------------+---------------------+---------------------------+--------------+
        |direction  |213772       |germany              |GERMANY                    |209643        |
        |direction  |213772       |france               |FRANCE                     |4129          |
        +-----------+-------------+---------------------+---------------------------+--------------+
    
    With this, you are now 100% sure that the change you wrote did not impact anything else... at least on this table.
    Let's say that you now recompute this table and use data-diff on the table downstream, and you notice that one
    of your tables (the one that generates the CEO's dashboard) has the following change:
    
        136935 (39.0%) rows are identical
        213772 (61.0%) rows have changed
        0 (0.0%) rows are only in 'left'
        0 (0.0%) rows are only in 'right
        
        Found the following changes:
        +------------+-------------+---------------------+---------------------------+--------------+
        |column_name |total_nb_diff|left_value           |right_value                |nb_differences|
        +------------+-------------+---------------------+---------------------------+--------------+
        |country_code|213772       |DE                   |NULL                       |209643        |
        |country_code|213772       |NULL                 |FR                         |4129          |
        +------------+-------------+---------------------+---------------------------+--------------+

    Now, that change is unexpected. So you go have a look at the SQL query generating this dashboard, and you notice this:
    
        SELECT
            ...
            CASE 
                WHEN country = "germany" THEN "DE"
                WHEN country = "FRANCE" THEN "FR
            END as country_code,
            ...
    
    Now it all makes sense! Perhaps the query used to be "correct" at some point in time when "germany" and "FRANCE"
    were the only possible values in that column, but this is not the case anymore. And by fixing one bug upstream, 
    you had unforeseen impacts on the downstream tables. Thankfully data-diff made it very easy to spot ! 
    
    !!! note 
        Of course, upon reading this, any senior analytics engineer is probably thinking that many things went wrong
        in the company to lead to this result. Indeed. Surely, the engineer who wrote that `CASE WHEN` statement should
        have performed some safety data harmonization in the cleaning stage of the pipeline. Surely, the developers of
        the data source never should have sent inconsistent data like this. But nevertheless that kind of scenario (and 
        much worse ones) often happens in real life, especially when you arrive in a young company that 
        "go fast and break things" and you inherit the legacy of your predecessors. 

### Implementing a new feature

Sometimes I am tasked with adding a new column, or enriching the content of an existing column. 
Once again, data-diff makes it very easy to make sure that I added the new column in the right place and that it 
contains the expected values.

### Reviewing other's changes

Code review is one of the most important engineering best practices of this century. I find that some data teams still
don't do it as much as they should, but we are getting there. dbt did a lot for the community to bring engineering
best-practices to the data teams. What makes code reviews even better and easier, is when a data-diff comes with
it. `DataDiffResults` can be exported as standalone HTML reports, which makes them very easy to share to others,
or to post in the comment of a Merge Request. If your DataOps are good enough, they can even automate the whole 
process and make your CI pipeline generate and post the data-diff report automatically for you.

If you prefer premium tech rather than building things yourself, I strongly suggest you have a look 
at [DataFold](https://www.datafold.com/) who provides on-the-shelf CI/CD for data teams, including a 
nice data-diff feature.

!!! note
    At this point, you are probably wondering why I went all the way to make my own data-diff tool, 
    if I recommend trying another paying tool that already does it on the shelf. Here a few elements of response:
    
    - Spark-frame is 100% free and open-source.
    - Datafold does have an open-source [data-diff](https://github.com/datafold/data-diff) version, but it is much more
      limited (and does not generate HTML reports). If I ever have the time, I will make a detailed feature comparison 
      of both data-diffs.
    - I believe spark-frame's data-diff is more powerful than Datafold's premium data-diff version, because it works
      well with complex data structures too.
    - I hope people will use spark-frame's (and [bigquery-frame](https://github.com/FurcyPin/bigquery-frame)'s) data-diff 
      to build up more nice features to easily have a full data-diff integration in their CI/CD pipelines. 


### See what changed in your data

When receiving full updates on a particular dataset. 
Data-diff can be simply used to display the changes that occurred between the old and new version of the data.

This can be useful to troubleshoot issues, or simply to know what changed.

However, this not the primary use-case we had in mind when making data-diff, and some other data-drift monitoring
tools might be better suited for this (like displaying the number of rows added per day, etc.).

Even though, advanced users might want to take a look at the 
[`DiffResult.diff_df_shards`][spark_frame.data_diff.DiffResult.diff_df_shards] attribute 
that provides access to the whole diff DataFrame, which can be used to retrieve the
exact set of rows that were updated, for instance.


### Releasing your data-models to prod like a pro

For me, the Graal :trophy: of analytics engineering best practices is to achieve full versioning of your data model.
Just like well-maintained libraries are fully versioned.

That would mean that every table you provide to your users has a version number. Whenever you introduce a breaking
change in the table's schema or data, you increase the minor version number (and you keep the major version for
full pipeline overhauls). You then maintain your pipelines for multiple versions, and leave the time for your users
to catch up with the changes before decommissioning the older versions.

Of course, this is quite difficult to achieve in practice because:

1. It's complicated to put in place:
     - SQL warehouses were not designed to facilitate this kind of logic[^1].
     - Most SQL development frameworks don't support this either[^2]. 
2. It can be very costly:
     - If your data warehouse already costs you an arm, then maintaining two or more versions of it would cost as many
       more limbs. This is clearly not feasible when you are trying to optimize costs.
       
[^1]: Which is a shame, really. I really hope that one day Spark and BigQuery will natively support version numbers for tables.
[^2]: Perhaps dbt will one day, or perhaps some dbt plugins will or already do. 
[sqlmesh](https://github.com/TobikoData/sqlmesh) does seem to go in that direction, which is nice.

One simpler alternative to versioning exists, it is called 
[Blue-Green deployment](https://dataintensivedreamer.medium.com/blue-green-deployment-in-a-data-architecture-aca64fd5e3c6).
It is a concept used in infrastructure deployment but the idea can be adapted to data pipelines.
The main advantage is that it limits the number of versions of your data to 2. But it means that you need 
your users to adapt to the new version quickly before being able to continue pushing new breaking changes.

Whichever strategy you choose, versioning or blue-green, I believe it would be extremely valuable for end users 
to get release notes whenever a model evolves, and if those notes showed you **exactly which table changed _and how_**. 
Not only for the table's _schema_ but also for the table's _data_. That would be, for me, the apex of analytics 
engineering best practices.



