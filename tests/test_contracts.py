"""
Tests for contract definitions and validation
"""

import pytest
from backend.contracts import (
    create_contract,
    create_continuous_contract,
    get_current_contract,
    get_contract_info,
    should_roll,
    CONTRACT_SPECS
)


def test_contract_specs_exist():
    """Test that all expected contract specs are defined"""
    assert 'MNQ' in CONTRACT_SPECS
    assert 'MES' in CONTRACT_SPECS
    assert 'MGC' in CONTRACT_SPECS


def test_create_contract_mnq():
    """Test creating MNQ contract"""
    contract = create_contract('MNQ', '202503')
    assert contract.symbol == 'MNQ'
    assert contract.exchange == 'CME'
    assert contract.currency == 'USD'
    assert contract.lastTradeDateOrContractMonth == '202503'


def test_create_contract_mes():
    """Test creating MES contract"""
    contract = create_contract('MES', '202503')
    assert contract.symbol == 'MES'
    assert contract.exchange == 'CME'
    assert contract.currency == 'USD'


def test_create_contract_mgc():
    """Test creating MGC contract with COMEX exchange"""
    contract = create_contract('MGC', '202502')
    assert contract.symbol == 'MGC'
    assert contract.exchange == 'COMEX'
    assert contract.currency == 'USD'


def test_create_contract_invalid_symbol():
    """Test that invalid symbol raises error"""
    with pytest.raises(ValueError, match="Unknown symbol"):
        create_contract('INVALID', '202503')


def test_create_continuous_contract():
    """Test creating continuous contract"""
    contract = create_continuous_contract('MNQ')
    assert contract.symbol == 'MNQ'
    assert contract.exchange == 'CME'


def test_get_current_contract():
    """Test getting current front month contract"""
    contract = get_current_contract('MNQ')
    assert contract.symbol == 'MNQ'
    assert len(contract.lastTradeDateOrContractMonth) == 6  # YYYYMM format


def test_get_contract_info():
    """Test getting contract specifications"""
    info = get_contract_info('MNQ')
    assert info['name'] == 'Micro E-mini Nasdaq-100'
    assert info['exchange'] == 'CME'
    assert info['multiplier'] == 2
    assert info['tick_size'] == 0.25
    assert info['tick_value'] == 0.50


def test_get_contract_info_all_symbols():
    """Test getting info for all supported symbols"""
    for symbol in ['MNQ', 'MES', 'MGC']:
        info = get_contract_info(symbol)
        assert 'name' in info
        assert 'exchange' in info
        assert 'multiplier' in info
        assert 'tick_size' in info


def test_should_roll_logic():
    """Test roll detection logic"""
    # Create a contract that's already expired (should roll)
    old_contract = create_contract('MNQ', '202401')  # January 2024
    assert should_roll(old_contract, days_before=8) == True

    # Current contract might or might not need rolling depending on date
    current = get_current_contract('MNQ')
    # Just test that it doesn't crash
    result = should_roll(current)
    assert isinstance(result, bool)


def test_contract_specs_completeness():
    """Test that all contract specs have required fields"""
    required_fields = ['name', 'exchange', 'currency', 'multiplier', 'tick_size', 'tick_value']

    for symbol, spec in CONTRACT_SPECS.items():
        for field in required_fields:
            assert field in spec, f"{symbol} missing {field}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
