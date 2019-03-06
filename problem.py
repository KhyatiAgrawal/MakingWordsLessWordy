import re

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry

@registry.register_problem
class VerbosityReduction(text_problems.Text2TextProblem):
  """Generate a concise sentence from a verbose sentence"""


  @property
  def approx_vocab_size(self):
    return 2**13  # ~8k

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 9,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split

    for (book_id, toskip) in books:
      text = cleanup.strip_headers(acquire.load_etext(book_id)).strip()
      lines = text.split("\n")[toskip:]
      prev_line = None
      ex_count = 0
      for line in lines:
        # Any line that is all upper case is a title or author name
        if not line or line.upper() == line:
          prev_line = None
          continue

        line = re.sub("[^a-z]+", " ", line.strip().lower())
        if prev_line and line:
          yield {
              "inputs": prev_line,
              "targets": line,
          }
          ex_count += 1
        prev_line = line