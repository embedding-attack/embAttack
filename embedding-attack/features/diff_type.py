import enum


class DiffType(enum.Enum):
    DIFFERENCE = ""
    DIFFERENCE_ONE_INIT = "_difference_one_init"
    RATIO = "_difftype_ratio"
    CONTINUE_TRAINING = "_continue_training"
    MOST_SIMILAR_EMBS_DIFF = "_most_similiar_embs_diff"
    MOST_SIMILAR_EMBS_DIFF_ONE_INIT = "_most_similiar_embs_diff_one_init"
    MOST_SIMILAR_EMBS_DIFF_ALL_EMBS = "_most_similiar_embs_diff_all_embs"
    MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE = "_most_similiar_embs_diff_one_init_continue"

    def has_one_init_graph(self):
        return  self in [DiffType.DIFFERENCE_ONE_INIT,
                          DiffType.MOST_SIMILAR_EMBS_DIFF,
                          DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT,
                          DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE]

    @staticmethod
    def get_list_of_all_diff_types():
        return [DiffType.DIFFERENCE,
                DiffType.DIFFERENCE_ONE_INIT,
                DiffType.MOST_SIMILAR_EMBS_DIFF,
                DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT,
                DiffType.MOST_SIMILAR_EMBS_DIFF_ALL_EMBS,
                DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE]

    @staticmethod
    def get_list_of_readable_diff_types(num_iter: int = None):
        '''
        names for diff types in "get_list_of_all_diff_types()'
        :return:
        '''
        if num_iter is None:
            n = 'n'
        else:
            n = str(num_iter)

        return {DiffType.DIFFERENCE.get_name(): f"a({n},{n},{n})",
                DiffType.DIFFERENCE_ONE_INIT.get_name(): f"a(1,{n},{n})",
                DiffType.MOST_SIMILAR_EMBS_DIFF.get_name(): f"s(1,{n},1)",
                DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT.get_name(): f"s(1,{n},{n})",
                DiffType.MOST_SIMILAR_EMBS_DIFF_ALL_EMBS.get_name(): f"s({n},{n},{n})",
                DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE.get_name(): f"sc(1,{n},{n})"}

    def __str__(self):
        return self.to_str()

    def set_iter(self, iter: int):
        self.iteration = iter
        return self

    def get_iter(self):
        if hasattr(self, 'iteration'):
            return self.iteration
        else:
            return -1

    def to_str(self):
        if hasattr(self, "iteration"):
            return self.value + f'_iter_{self.iteration}'
        else:
            return self.value

    def has_iteration(self):
        return hasattr(self, "iteration")

    def get_name(self, num_iter:int=None):
        return self.value


if __name__ == '__main__':
    dt = DiffType.MOST_SIMILAR_EMBS_DIFF

    print(dt.has_one_init_graph())
    print(dt.name)
