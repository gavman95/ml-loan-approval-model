### Test main file

from main import main


def test_main_prints_greeting(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello from mlopsbootcamp!" in captured.out
