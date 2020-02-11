import argparse
import pandas as pd
from pathlib import Path

# all extra wells are control wells


def get_total_number_of_wells(all_plates_translation_csv):
    number_of_considered_well_per_plate = 4 * 12 # 4 because we are just looking at the first and last 2 rows
    total_number_of_wells = len(all_plates_translation_csv) * number_of_considered_well_per_plate
    return total_number_of_wells

def get_all_control_wells(all_plates_translation_csv):
    all_control_wells = []
    for well_split_dir in all_plates_translation_csv["anonymous_plate_path_dir_well_split"]:
        all_control_wells.append(well_split_dir / "H11.png")
        all_control_wells.append(well_split_dir / "H12.png")
    return all_control_wells

def get_all_non_control_wells(all_plates_translation_csv):
    all_non_control_wells = []
    for well_split_dir in all_plates_translation_csv["anonymous_plate_path_dir_well_split"]:
        for letter in "ABCDEFGH":
            for number in range(1, 13):
                if (letter, number) != ("H", 11) and (letter, number) != ("H", 12):
                    all_non_control_wells.append(well_split_dir / f"{letter}{number}.png")
    return all_non_control_wells

def make_participants_datasets(number_of_participants, number_of_control_wells_per_part,
                               number_of_non_control_wells_per_part, number_of_shared_control_wells_per_part,
                               all_control_wells, all_non_control_wells):
    participant_index_to_list_of_its_wells = [] * number_of_participants

    nb_of_control_wells_before = len(all_control_wells)
    add_shared_control_wells_for_pairs_of_participants(participant_index_to_list_of_its_wells,
                                                       all_control_wells,
                                                       number_of_shared_control_wells_per_part)
    nb_of_control_wells_after = len(all_control_wells)
    assert nb_of_control_wells_before == nb_of_control_wells_after + (number_of_participants * number_of_shared_control_wells_per_part)

    add_control_wells_for_each_participant(participant_index_to_list_of_its_wells,
                                           all_control_wells,
                                           number_of_control_wells_per_part)
    nb_of_control_wells_after = len(all_control_wells)
    assert nb_of_control_wells_before == nb_of_control_wells_after + (
                number_of_participants * number_of_control_wells_per_part)

    nb_of_non_control_wells_before = len(all_non_control_wells)
    add_non_control_wells_for_each_participant(participant_index_to_list_of_its_wells,
                                           all_non_control_wells,
                                           number_of_non_control_wells_per_part)
    nb_of_non_control_wells_after = len(all_non_control_wells)
    assert nb_of_non_control_wells_before == nb_of_non_control_wells_after + (number_of_participants * number_of_non_control_wells_per_part)

    participant_index_to_list_of_its_extra_control_wells = [] * number_of_participants
    add_extra_control_wells_per_participant(participant_index_to_list_of_its_extra_control_wells,
                                            all_control_wells)
    assert len(all_control_wells) == 0

    return participant_index_to_list_of_its_wells, participant_index_to_list_of_its_extra_control_wells


def create_participants_datasets_in_disk(participant_index_to_list_of_its_wells, participant_index_to_list_of_its_extra_control_wells,
                                         output_dir):
    pass


def create_participants_dataset(all_plates_translation_csv_filepath, number_of_participants,
                                nb_of_wells_per_part,
                                percentage_of_images_from_control_wells,
                                percentage_of_shared_control_well_between_participants,
                                output_dir):
    all_plates_translation_csv = pd.read_csv(all_plates_translation_csv_filepath)
    total_number_of_wells = get_total_number_of_wells(all_plates_translation_csv)
    max_number_of_wells_per_part = int(total_number_of_wells / number_of_participants)

    nb_of_wells_per_part = min(nb_of_wells_per_part, max_number_of_wells_per_part)

    number_of_control_wells_per_part = int((percentage_of_images_from_control_wells/100)*nb_of_wells_per_part)
    assert 0 <= number_of_control_wells_per_part <= nb_of_wells_per_part

    number_of_non_control_wells_per_part = nb_of_wells_per_part - number_of_control_wells_per_part
    assert 0 <= number_of_non_control_wells_per_part <= number_of_control_wells_per_part

    assert number_of_control_wells_per_part + number_of_non_control_wells_per_part == number_of_control_wells_per_part

    number_of_shared_control_wells_per_part = int((percentage_of_shared_control_well_between_participants/100)*number_of_control_wells_per_part)

    all_control_wells = get_all_control_wells(all_plates_translation_csv)
    all_non_control_wells = get_all_non_control_wells(all_plates_translation_csv)
    participant_index_to_list_of_its_wells, participant_index_to_list_of_its_extra_control_wells =\
        make_participants_datasets(number_of_participants, number_of_control_wells_per_part,
                               number_of_non_control_wells_per_part, number_of_shared_control_wells_per_part,
                               all_control_wells, all_non_control_wells)

    create_participants_datasets_in_disk(participant_index_to_list_of_its_wells, participant_index_to_list_of_its_extra_control_wells,
                                         output_dir)





def get_args():
    parser = argparse.ArgumentParser(description='Create participants datasets.')
    parser.add_argument('--all_plates_translation_csv_filepath', type=str, help='all_plates.translation.csv output by create_split_well_images.py script', required=True)
    parser.add_argument('--nb_part', type=int, help='Number of participants', required=True)
    parser.add_argument('--nb_of_wells_per_part', type=int, help='Number of wells per participant', required=True)
    parser.add_argument('--control_well', type=float, help='Percentage of images that should come from control wells', required=True)
    parser.add_argument('--share_control_well', type=float, help='Percentage of control wells between pair of participants',
                        required=True)
    parser.add_argument('--output_dir', type=str, help='Directory to output the participants dataset', required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    all_plates_translation_csv_filepath = Path(args.all_plates_translation_csv_filepath)
    number_of_participants = args.nb_part
    percentage_of_images_from_control_wells = args.control_well
    percentage_of_shared_control_well_between_participants = args.share_control_well
    nb_of_wells_per_part = args.nb_of_wells_per_part
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True)
    create_participants_dataset(all_plates_translation_csv_filepath, number_of_participants,
                                nb_of_wells_per_part,
                                percentage_of_images_from_control_wells,
                                percentage_of_shared_control_well_between_participants,
                                output_dir)

