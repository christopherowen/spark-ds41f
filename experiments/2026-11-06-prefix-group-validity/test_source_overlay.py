"""Reject mixed source/image state before a serving comparison."""
import copy
import unittest

from source_overlay import validate_snapshot


class SourceOverlayTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            'parent_image_id': 'sha256:parent',
            'files': [{'container_path': '/pkg/model.py', 'export': 'experiment/model.py',
                       'candidate_sha256': 'new', 'control_sha256': 'old'}],
        }
        self.candidate = {
            'image_id': 'sha256:parent', 'hashes': {'/pkg/model.py': 'new'},
            'mounts': [{'Destination': '/pkg/model.py', 'Source': '/repo/experiment/model.py', 'RW': False}],
        }

    def test_every_candidate_file_must_be_from_the_published_read_only_export(self):
        validate_snapshot(self.manifest, 'candidate', self.candidate)
        for field, value in [('RW', True), ('Source', '/tmp/unpublished/model.py')]:
            bad = copy.deepcopy(self.candidate)
            bad['mounts'][0][field] = value
            with self.assertRaises(RuntimeError):
                validate_snapshot(self.manifest, 'candidate', bad)
        bad = copy.deepcopy(self.candidate)
        bad['mounts'] = []
        with self.assertRaises(RuntimeError):
            validate_snapshot(self.manifest, 'candidate', bad)

    def test_a_matching_tag_does_not_cover_wrong_image_or_source(self):
        for key, value in [('image_id', 'sha256:different'), ('hashes', {'/pkg/model.py': 'old'})]:
            bad = {**self.candidate, key: value}
            with self.assertRaises(RuntimeError):
                validate_snapshot(self.manifest, 'candidate', bad)

    def test_control_cannot_keep_candidate_mounts(self):
        control = {**self.candidate, 'hashes': {'/pkg/model.py': 'old'}, 'mounts': []}
        validate_snapshot(self.manifest, 'control', control)
        control['mounts'] = self.candidate['mounts']
        with self.assertRaises(RuntimeError):
            validate_snapshot(self.manifest, 'control', control)


if __name__ == '__main__':
    unittest.main()
